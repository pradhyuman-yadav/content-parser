import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from guardrails import Guard
from pydantic import BaseModel, Field


try:
    client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
except KeyError:
    raise KeyError("GOOGLE_API_KEY not found in environment variables. Please create a .env file and add it.")

def call_gemini_with_prompt(model, messages, **kwargs) -> str:
    """
    A wrapper function to call the Gemini API with a given prompt and configuration.
    This makes the Gemini API compatible with the Guardrails library's expectations.
    """
    try:

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=500,
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
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        # Return an empty string or handle the error as appropriate
        return ""

# --- 2. DEFINE OUTPUT STRUCTURE WITH PYDANTIC ---

class EmailType(BaseModel):
    proccess: bool = Field(False, description="Check if the email is a valid contract related email or not.")

##############

class Location(BaseModel):
    city: Optional[str] = Field(None, description="The city of pickup or delivery.")
    state: Optional[str] = Field(None, description="The state's two-letter abbreviation.")
    zip_code: Optional[str] = Field(None, description="The zip code.")

class DimensionDetail(BaseModel):
    length: Optional[float] = Field(None, description="The length of the dimension.")
    width: Optional[float] = Field(None, description="The width of the dimension.")
    height: Optional[float] = Field(None, description="The height of the dimension.")

class Others(BaseModel):
    piece_count: Optional[int] = Field(None, description="The number of pieces or pallets.")
    stackable: Optional[str] = Field(None, description="Whether the freight is stackable.")
    hazardous: Optional[bool] = Field(None, description="Whether the shipment is hazardous.")
    weight_lbs: Optional[float] = Field(None, description="The total weight in pounds.")
    us_vehicle: Optional[str] = Field(None, description="The type of US vehicle required.")
    mx_vehicle: Optional[str] = Field(None, description="The type of MX vehicle required.")
    requirements: Optional[str] = Field(None, description="Any other special requirements.")

class Shipment(BaseModel):
    pickup: Optional[List[Location]] = Field(None, description="A list of pickup locations.")
    delivery: Optional[List[Location]] = Field(None, description="A list of delivery locations.")
    pickup_eta: Optional[List[str]] = Field(None, description="The estimated date or time for multiple listed pickups.")
    delivery_eta: Optional[List[str]] = Field(None, description="The estimated date or time for multiple listed delivery.")
    miles: Optional[float] = Field(None, description="The total distance in miles listed.")
    travel_time: Optional[str] = Field(None, description="The total time taken or hours spent for the delivery.")
    dimensions: Optional[List[DimensionDetail]] = Field(None, description="A list of dimensions for each piece in the shipment.")
    others: Optional[Others] = None

class Rates(BaseModel):
    base_rate: Optional[float] = Field(None, description="The base rate for the shipment.")
    starboard_rates: Optional[List[str]] = Field(None, description="A list of supplementary rates from Starboard.")

class ExpectedOutput(BaseModel):
    language: Optional[str] = Field(None, description="The primary language of the source text (e.g., English, Spanish).")
    shipment: Optional[Shipment] = None
    rates: Optional[Rates] = None

# --- 3. MAIN FUNCTION ---

def process_text_with_validators(text_data: str) -> Dict[str, Any]:
    """
    Uses a two-step Guardrails process to first validate structure, then provenance.
    """

    prompt = f"""
    Based *only* on the text provided below, extract the information as a JSON object that strictly follows the required schema.
    If a value is not found in the text, use `null` only for optional fields.

    --- TEXT ---
    {text_data}
    --- END TEXT ---

    ${{gr.complete_json_suffix_v2}}
    """
    
    try:
        
        print("Checking for relavance")
        # Step 0: Check for email
        check_email = Guard.for_pydantic(
            output_class=EmailType
        )
        response_email = check_email(
            call_gemini_with_prompt,
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a expert in email classification. Classify the following email text. If it is a contract or a communication email, return True. Otherwise, if it is a spam or advertisement email, return False. Only return True or False."},
                {"role": "user", "content": prompt}
            ],
        )
        validated_email = response_email.validated_output

        if not validated_email.get('proccess'):
            return "Not a relavent email."


        # --- Step 1: Validate JSON structure and schema ---
        structure_guard = Guard.for_pydantic(
            output_class=ExpectedOutput
        )

        print("Calling for structured extraction...")
        response = structure_guard(
            call_gemini_with_prompt,
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a data extraction expert. You only respond with the single piece of requested information and nothing else. If the information is not found, you respond with 'N/A'."},
                {"role": "user", "content": prompt}
            ],
            
        )
        
        validated_structure = response.validated_output
        # raw_output = response.raw_llm_output
        
        if not validated_structure:
            raise ValueError("Structural validation failed. The LLM output did not match the Pydantic schema.")
        
        # print("Validating extracted fields against source text for provenance...")
        # is_provenance_valid, failed_fields = validate_fields_against_source(validated_structure, text_data)

        # if not is_provenance_valid:
        #     raise ValueError(f"Provenance validation failed. The following fields could not be found in the source text: {failed_fields}")

        # --- Step 3: If both checks pass, return the result ---
        print("✅ SUCCESS: Both structure and provenance checks passed?.")
        
        return validated_structure

    except Exception as e:
        print("❌ GUARDRAIL FAILED: The process did not pass validation.")
        print(f"   - Reason: {e}")
        return {"error": "Guardrails validation failed.", "details": str(e)}
    
def validate_fields_against_source(
    validated_data: Union[Dict, BaseModel],
    source_text: str
) -> Tuple[bool, List[str]]:
    """
    Recursively iterates through a validated Pydantic model or dictionary
    and checks if each string value exists in the source text.

    Args:
        validated_data: The validated data (Pydantic model or dictionary).
        source_text: The original text to check against.

    Returns:
        A tuple containing a boolean (True if all fields are valid) and a
        list of fields that were not found in the source text.
    """
    unverified_fields = []

    # If the input is a Pydantic model, convert it to a dictionary
    if isinstance(validated_data, BaseModel):
        data_dict = validated_data.dict()
    else:
        data_dict = validated_data

    for key, value in data_dict.items():
        # Custom rule: Skip provenance check for the 'language' field as it's inferred.
        if key == 'language':
            continue

        if value is None:
            continue

        if isinstance(value, dict):
            is_valid, nested_unverified = validate_fields_against_source(value, source_text)
            if not is_valid:
                unverified_fields.extend(nested_unverified)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    is_valid, nested_unverified = validate_fields_against_source(item, source_text)
                    if not is_valid:
                        unverified_fields.extend(nested_unverified)
                # We only check string items for provenance
                elif isinstance(item, str):
                    if item.lower() not in source_text.lower():
                        unverified_fields.append(item)
        # Check for string types to validate against the source
        elif isinstance(value, str):
            # A simple 'in' check is a direct way to verify provenance
            if value.lower() not in source_text.lower():
                unverified_fields.append(value)
    
    return not unverified_fields, unverified_fields


# --- 4. EXAMPLE USAGE ---
# if __name__ == "__main__":
#     sample_text = """
#     Subject: RATE REQUEST EXPEDITED: Oportunidad Cross Border / Pierburg Pump Mexico
 
#     Buen día,
    
#     Me apoyan con las siguientes cotizaciones de unidades dedicadas y expeditadas
    
#     Shipper : Pierburg Pump Technology México, S.A. de C.V. →
#     Cnee:  Falcon Manufacturing LLC
    
#     Ruta: Celaya, Gto. México → Columbus, IN, EE.UU.
#     Servicio: Dedicado Express Puerta a Puerta
#     Mercancía: Partes automotrices (sin mercancía peligrosa - No DGR)
    
#     1. Detalles de los Pallets
#     Tipo de Pallet   Dimensiones (cm)     	Peso Bruto (kg) Volumen (m³)
#     Tipo 1   130 x 120 x 100          	470.0	1.56
#     Tipo 2   130 x 120 x 100          	238.0	1.56
    
#     2. Frecuencia Estimada
#     Promedio diario: 3 pallets por día
    
#     Máximo por envío: hasta 10 pallets Tipo 1 y 15 pallets Tipo 2 cada 3 dias aprox
    
#     Vigencia estimada del proyecto: Diciembre 2025
    
#     Propuesta de Precios
    
#     Precio por camión expeditado cruce incluido
    
#     Otras Consideraciones
#     Certificaciones de cumplimiento disponibles bajo solicitud (CTPAT, OEA, etc.)
    
#     Saludos / Best Regards,

#     """
#     print("Starting high-efficiency data extraction process...")
#     extracted_data = process_text_with_validators(sample_text)
#     print("\n=============================================")
#     print("✅ Extraction Complete. Final Validated JSON")
    
#     # Save the output to a timestamped JSON file
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"GP_extracted_data_{timestamp}.json"
    
#     output_dict = extracted_data
#     if hasattr(extracted_data, 'dict'):
#         output_dict = extracted_data.dict()

#     with open(filename, "w") as file:
#         json.dump(output_dict, file, indent=2)
    
#     print(f"Output saved to {filename}")