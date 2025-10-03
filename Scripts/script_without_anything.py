from datetime import datetime
import os
import json
from openai import OpenAI
from typing import Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. CONFIGURE OPENAI CLIENT ---
# The client will automatically look for the OPENAI_API_KEY environment variable.
# Handle the case where the key is not set.
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure your OPENAI_API_KEY environment variable is set correctly.")
    client = None

# --- 2. CORE EXTRACTION AND VALIDATION LOGIC ---

def get_llm_extraction(text_data: str, field_description: str) -> str:
    """
    Makes a single, focused API call to OpenAI to extract a specific field.
    
    Args:
        text_data: The source text to search within.
        field_description: A human-readable description of the data to find.
        
    Returns:
        The extracted information as a string.
    """
    prompt = f"""
    Based *only* on the text provided below, extract the value for: '{field_description}'.
    Respond with ONLY the extracted value and nothing else.
    If the information cannot be found in the text, respond with the exact string "N/A".

    --- TEXT ---
    {text_data}
    --- END TEXT ---

    Value for '{field_description}':
    """
    try:
        response = client.chat.completions.create(
            # Using gpt-4 for higher accuracy, but gpt-3.5-turbo can be used for speed/cost.
            model="gpt-5", 
            messages=[
                {"role": "system", "content": "You are a highly precise data extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Set to 0 for deterministic, factual responses
            max_tokens=60   # Limit the output size
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during the OpenAI API call: {e}")
        return "N/A"

def extract_and_validate_field(
    text_data: str, 
    field_description: str, 
    expected_type: str, 
    max_retries: int = 3
) -> Union[str, int, float, bool, None]:
    """
    Extracts a field, validates it against the source text (guardrail), and handles retries.
    """
    for attempt in range(max_retries):
        extracted_value = get_llm_extraction(text_data, field_description)

        if extracted_value == "N/A":
            print(f"INFO: '{field_description}' not found in text. Returning None.")
            return None

        # --- GUARDRAIL & FACT-CHECK ---
        # Verify that the extracted information is present in the original text.
        # This is a crucial step to prevent hallucinations.
        # We use lowercase for a more robust comparison.
        if extracted_value.lower() in text_data.lower():
            print(f"✅ SUCCESS: Extracted '{extracted_value}' for '{field_description}'. Validation passed.")
            
            # --- Type Conversion ---
            try:
                if expected_type == "number":
                    cleaned_val = extracted_value.replace(",", "").replace("$", "")
                    return float(cleaned_val) if '.' in cleaned_val else int(cleaned_val)
                elif expected_type == "boolean":
                    return "yes" in extracted_value.lower() or "true" in extracted_value.lower() or "stackable" in extracted_value.lower()
                else: # string
                    return extracted_value
            except ValueError:
                print(f"⚠️ WARN: Could not convert '{extracted_value}' to type '{expected_type}'. Retrying...")
                continue # Retry if type conversion fails

        else:
            print(f"❌ GUARDRAIL FAILED: Extracted '{extracted_value}' for '{field_description}' not found in source text. Retrying ({attempt + 1}/{max_retries})...")
    
    print(f"🚨 ERROR: Failed to reliably extract '{field_description}' after {max_retries} attempts.")
    return None

# --- 3. MAIN FUNCTION ---

def process_text_to_json(text_data: str) -> Dict[str, Any]:
    """
    Takes raw text data, extracts structured information field-by-field using OpenAI
    with validation guardrails, and returns a JSON-compatible dictionary.
    
    Args:
        text_data: A string containing the unstructured information.
        
    Returns:
        A dictionary populated with the extracted data, matching the desired format.
    """
    if not client:
        raise ConnectionError("OpenAI client is not initialized. Check your API key.")

    # The fixed JSON format that defines the structure and expected data types.
    json_schema = {
      "shipment": {
        "pickup": {"city": "string", "state": "string", "zip_code": "string"},
        "delivery": {"city": "string", "state": "string", "zip_code": "string"},
        "pickup_eta": "string", "delivery_eta": "string", "hours": "string",
        "pickup_timezone": "string", "delivery_timezone": "string", "miles": "number",
        "dimensions": {"dimension_1": {"length": "number", "width": "number", "height": "number"}},
        "others": {
          "piece_count": "number", "stackable": "string", "hazardous": "boolean",
          "weight_lbs": "number", "us_vehicle": "string", "mx_vehicle": "string",
          "requirements": "string"
        }
      },
      "rates": {"base_rate": "number", "starboard_rates": {"rate_1": "string"}}
    }

    output_data = {}

    # Helper function to recursively traverse the schema and populate the output_data dict.
    def populate_data(schema_level, output_level, path_prefix=""):
        for key, value in schema_level.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            
            if isinstance(value, dict):
                output_level[key] = {}
                populate_data(value, output_level[key], current_path)
            else:
                expected_type = value
                # Create a user-friendly description for the LLM prompt.
                field_description = current_path.replace("_", " ").replace(".", " -> ")
                print(f"\n--- Extracting: {field_description} ---")
                
                extracted_value = extract_and_validate_field(
                    text_data=text_data,
                    field_description=field_description,
                    expected_type=expected_type
                )
                output_level[key] = extracted_value

    populate_data(json_schema, output_data)
    return output_data

# --- 4. EXAMPLE USAGE ---

if __name__ == "__main__":
    # Sample text data containing shipment information.
    sample_text = """
        
    Origin City, State, Zip	Concord Township, OH 44077
    Ready Date/Time	 Tmrw
    Destination City, State, Zip	Ridgway PA 15853
    Delivery Date/Time	Direct/ or next day
    Pallet/Skid Count	1 skid
    Total Weight	1100#
    Dimensions – LxWxH (Turnable?)	66" Lx 48” W x 60" T. Yes, can be turned
    Stackable or Non-Stack?	 
    Is a Dock High Vehicle Required?  If so, do you need True Dock High or are ramps acceptable?	 Can be loaded from ground

    """
    
    print("Starting data extraction process...")
    extracted_data = process_text_to_json(sample_text)
    
    print("\n=============================================")
    print("✅ Extraction Complete. Final JSON Output:")
    print("=============================================")
    
    # Use json.dumps for a nicely formatted string output.
    # print(json.dumps(extracted_data, indent=2))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"extracted_data_{timestamp}.json"
    with open(filename, "w") as file:
        json.dump(extracted_data, file, indent=2)