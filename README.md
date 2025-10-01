# High-Accuracy Data Extraction with Provenance Validation

This script demonstrates a robust method for extracting structured data from unstructured text using a Large Language Model (LLM) and validating the results for factual consistency. It processes text field-by-field based on a predefined JSON schema, ensuring that each extracted piece of information is traceable to the source text.

## Overview

The primary goal of this script is to perform high-accuracy, validated data extraction from text documents, such as emails or reports. It uses the OpenAI GPT-4 model for extraction and the `Guardrails` library with a `ProvenanceLLM` validator to ensure that the extracted data is factually grounded in the provided source text.

The script is designed to be modular and configurable, allowing users to easily define the data they want to extract by modifying a JSON schema.

## Features

-   **Field-by-Field Extraction**: Extracts data for each field defined in a schema individually for higher accuracy.
-   **Provenance Validation**: Uses `guardrails.hub.ProvenanceLLM` to verify that the extracted information is present in the source text, reducing hallucinations.
-   **Schema-Driven**: A simple JSON schema defines the structure of the data to be extracted.
-   **Handles Complex Structures**: Capable of extracting nested objects and lists of strings or objects.
-   **Self-Correction**: The `Guardrails` setup is configured to attempt to "fix" any data that fails the initial validation.
-   **Clear Logging**: Provides detailed console output for each step of the extraction and validation process.
-   **JSON Output**: Saves the final validated data into a timestamped JSON file.

## Requirements

-   Python 3.7+
-   An OpenAI API Key
-   The following Python libraries:
    -   `openai`
    -   `guardrails-ai`
    -   `sentence-transformers`
    -   `numpy`
    -   `python-dotenv`

## Setup

1.  **Clone the repository or download the script.**

2.  **Install the required Python libraries:**

    ```bash
    pip install openai "guardrails-ai>=0.4.0" sentence-transformers numpy python-dotenv
    ```

3.  **Create a `.env` file** in the same directory as the script and add your OpenAI API key:

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Usage

1.  **Configure the Schema**: Open `script_with_ProvenanceLLM.py` and modify the `json_schema` dictionary inside the `process_text_field_by_field` function to match the data you wish to extract.

2.  **Set the Input Text**: In the `if __name__ == "__main__"` block, replace the `sample_text` variable with the text you want to process.

3.  **Run the script from your terminal:**

    ```bash
    python script_with_ProvenanceLLM.py
    ```

The script will print the extraction progress to the console and create a new JSON file (e.g., `GPyP_extracted_data_20250930_171003.json`) with the validated data.

## How It Works

1.  **Initialization**: The script loads environment variables, initializes the OpenAI client, and loads the `paraphrase-MiniLM-L6-v2` sentence-embedding model required for semantic validation.

2.  **Schema Iteration**: The `process_text_field_by_field` function recursively iterates through the `json_schema`.

3.  **Extraction**: For each field, it constructs a targeted prompt asking the LLM to extract only that specific piece of information.
    -   For simple fields, it calls `extract_and_validate_field`.
    -   For lists, it calls `extract_and_validate_list`.

4.  **Validation**: The `Guardrails` library wraps the LLM call. The `ProvenanceLLM` validator takes the LLM's output and compares it against the source text. It uses a sentence transformer model to semantically verify if the extracted statement can be considered a fact present in the source.

5.  **Output Generation**: If validation is successful, the data is added to the final `output_data` dictionary. If a field is not found, it is stored as `None`. The final dictionary is then written to a JSON file.

## Customizing the Schema

The `json_schema` is a Python dictionary that defines the desired output structure.

-   **Simple Field**: A key-value pair where the value is a string describing the field.
    ```python
    "language": "Language of the text"
    ```

-   **Nested Object**: A key-value pair where the value is another dictionary.
    ```python
    "shipment": {
        "pickup": {"city": "Pickup City", "state": "Pickup State"}
    }
    ```

-   **List of Objects**: A key with a list containing a single example object. The keys of the object define the structure of each item in the list.
    ```python
    "dimensions": [{"length": "Length", "width": "Width", "height": "Height"}]
    ```

-   **List of Strings**: A key with a list containing a single example string.
    ```python
    "starboard_rates": ["Starboard Rate"]
    ```

## Output

The script generates a JSON file with a name like `GPyP_extracted_data_YYYYMMDD_HHMMSS.json`. This file contains the structured data extracted and validated from the source text.

**Example Output Snippet:**
```json
{
  "language": "Spanish",
  "shipment": {
    "pickup": {
      "city": "Celaya",
      "state": "Gto",
      "zip_code": null
    },
    "delivery": {
      "city": "Columbus",
      "state": "IN",
      "zip_code": null
    },
    // ... more data
  }
}
```
