import os
import json
from typing import Dict, Any, Union
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Guardrails and the ProvenanceLLM validator
import guardrails as gd
from guardrails.hub import ProvenanceLLM
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "This script requires `sentence-transformers`. Install it with `pip install sentence-transformers`."
    )

# --- 1. SETUP CLIENTS AND MODELS ---

try:
    # client = genai.Client(
    #     api_key=os.environ.get("GEMINI_API_KEY"),
    # )
    client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
except KeyError:
    raise KeyError("GOOGLE_API_KEY not found in environment variables. Please create a .env file and add it.")

def call_gemini_with_prompt(messages, **kwargs) -> str:
    """
    A wrapper function to call the Gemini API with a given prompt and configuration.
    This makes the Gemini API compatible with the Guardrails library's expectations.
    """
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            temperature=0,
            max_tokens=50,
            extra_body={
            'extra_body': {
                "google": {
                "thinking_config": {
                    "thinking_budget": 0,
                    "include_thoughts": False
                }
                }
            }
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        # Return an empty string or handle the error as appropriate
        return ""

# Load embedding model for validation
print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")
print("Model loaded.")

def embed_function(sources: list[str]) -> np.array:
    return EMBEDDING_MODEL.encode(sources)

# --- 2. CORE EXTRACTION AND VALIDATION LOGIC ---

def extract_and_validate_field(
    text_data: str,
    field_description: str,
) -> Union[str, None]:
    """
    Extracts a single field and validates it against the source text using ProvenanceLLM.
    """
    print(f"\n--- Extracting: {field_description} ---")

    # Guard configured to only check for factual consistency (provenance)
    guard = gd.Guard().use(
        ProvenanceLLM(
            on_fail="fix",
        )
    )

    # A focused prompt for extracting just one piece of information
    messages = [
        {"role": "system", "content": "You are a data extraction expert. You only respond with the single piece of requested information and nothing else. If the information is not found, you respond with 'N/A'."},
        {"role": "user", "content": f"Based *only* on the text provided below, what is the value for '{field_description}'?\n\n--- TEXT ---\n{text_data}\n--- END TEXT ---"}
    ]

    try:
        response = guard(
            call_gemini_with_prompt,
            messages=messages,
            metadata={
                "sources": [text_data],
                "embed_function": embed_function,
            },
        )

        extracted_value = response.validated_output
        if extracted_value == "N/A":
            print(f"INFO: '{field_description}' not found. Returning None.")
            return None

        print(f"✅ SUCCESS: Extracted and validated '{extracted_value}' for '{field_description}'.")
        return extracted_value

    except Exception as e:
        print(f"❌ GUARDRAIL FAILED for '{field_description}'. Reason: {e}")
        return None

def extract_and_validate_list(text_data: str, list_description: str, item_schema: Any) -> list:
    """
    Extracts a list of items (either strings or objects) and validates the raw output.
    """
    print(f"\n--- Extracting List: {list_description} ---")

    # Build the prompt based on whether we expect objects or strings
    if isinstance(item_schema, dict):
        schema_str = json.dumps(item_schema)
        prompt_instruction = f"Return the result as a JSON list of objects. Each object should have keys like in this example: {schema_str}."
    else:
        prompt_instruction = "Return the result as a JSON list of strings."

    final_prompt = f"""
    Based *only* on the text provided below, extract all items for '{list_description}'.
    {prompt_instruction}
    If no items are found, return an empty list: [].

    --- TEXT ---
    {text_data}
    --- END TEXT ---
    """

    messages = [
        {"role": "system", "content": "You are a data extraction expert. You only respond with a valid JSON list."},
        {"role": "user", "content": final_prompt}
    ]

    guard = gd.Guard().use(ProvenanceLLM(on_fail="fix"))

    try:
        response = guard(
            call_gemini_with_prompt,
            messages=messages,
            metadata={"sources": [text_data], "embed_function": embed_function}
        )
        raw_output = response.raw_llm_output
        
        # We need to parse the raw string output into a Python list
        try:
            parsed_list = json.loads(raw_output)
            if isinstance(parsed_list, list):
                print(f"✅ SUCCESS: Extracted and validated list for '{list_description}'.")
                return parsed_list
            else:
                print(f"⚠️ WARN: LLM did not return a list for '{list_description}'. Got {type(parsed_list)}. Returning empty list.")
                return []
        except json.JSONDecodeError:
            print(f"⚠️ WARN: Could not parse LLM output as JSON for '{list_description}'. Returning empty list.")
            return []

    except Exception as e:
        print(f"❌ GUARDRAIL FAILED for list '{list_description}'. Reason: {e}")
        return [] # Return empty list on failure to ensure key is present
# --- 3. MAIN FUNCTION ---

def process_text_field_by_field(text_data: str) -> Dict[str, Any]:
    """
    Takes raw text and extracts structured information field-by-field,
    validating each with Guardrails.
    """

    # The schema defines what we want to extract, including special formats for lists.
    json_schema = {
        "language": "Language of the text",
        "shipment": {
            "pickup": {"city": "Pickup City", "state": "Pickup State", "zip_code": "Pickup Zip Code"},
            "delivery": {"city": "Delivery City", "state": "Delivery State", "zip_code": "Delivery Zip Code"},
            "pickup_eta": "Pickup Date or ETA",
            "delivery_eta": "Delivery Date or ETA",
            "miles": "Total Miles",
            "Hours": "Time taken or Hours spent for Delivery",
            # Special format: A list containing one example object defines a list of objects
            "dimensions": [{"length": "Length", "width": "Width", "height": "Height"}],
            "others": {
                "piece_count": "Piece Count",
                "stackable": "Stackable",
                "hazardous": "Is Hazardous",
                "weight_lbs": "Weight in Pounds",
                "us_vehicle": "US Vehicle",
                "mx_vehicle": "MX Vehicle",
                "requirements": "Requirements"
            }
        },
        "rates": {
            "base_rate": "Base Rate",
            # Special format: A list containing one example string defines a list of strings
            "starboard_rates": ["Starboard Rate"]
        }
    }

    output_data = {}

    def populate_data(schema_level, output_level):
        for key, value in schema_level.items():
            if isinstance(value, list) and value:
                # Handle list extraction
                item_schema = value[0]
                list_description = f"List of {key.replace('_', ' ')}"
                output_level[key] = extract_and_validate_list(text_data, list_description, item_schema)
            elif isinstance(value, dict):
                output_level[key] = {}
                populate_data(value, output_level[key])
            else:
                field_description = value
                output_level[key] = extract_and_validate_field(text_data, field_description)

    populate_data(json_schema, output_data)
    return output_data

# --- 4. EXAMPLE USAGE ---
if __name__ == "__main__":
    sample_text = """
Subject: RATE REQUEST EXPEDITED: Oportunidad Cross Border / Pierburg Pump Mexico
 
Buen día,
 
Me apoyan con las siguientes cotizaciones de unidades dedicadas y expeditadas
 
Shipper : Pierburg Pump Technology México, S.A. de C.V. →
Cnee:  Falcon Manufacturing LLC
 
Ruta: Celaya, Gto. México → Columbus, IN, EE.UU.
Servicio: Dedicado Express Puerta a Puerta
Mercancía: Partes automotrices (sin mercancía peligrosa - No DGR)
 
1. Detalles de los Pallets
Tipo de Pallet   Dimensiones (cm)     	Peso Bruto (kg) Volumen (m³)
Tipo 1   130 x 120 x 100          	470.0	1.56
Tipo 2   130 x 120 x 100          	238.0	1.56
 
2. Frecuencia Estimada
Promedio diario: 3 pallets por día
 
Máximo por envío: hasta 10 pallets Tipo 1 y 15 pallets Tipo 2 cada 3 dias aprox
 
Vigencia estimada del proyecto: Diciembre 2025
 
Propuesta de Precios
 
Precio por camión expeditado cruce incluido
 
Otras Consideraciones
Certificaciones de cumplimiento disponibles bajo solicitud (CTPAT, OEA, etc.)
 
Saludos / Best Regards,
    """
    print("Starting high-accuracy data extraction process...")
    extracted_data = process_text_field_by_field(sample_text)
    print("\n=============================================")
    print("✅ Extraction Complete. Final Validated JSON:")
    # print(json.dumps(extracted_data, indent=2))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"GPyP_extracted_data_{timestamp}.json"
    with open(filename, "w") as file:
        json.dump(extracted_data, file, indent=2)
