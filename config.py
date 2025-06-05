import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

# FILE PATH
EXCEL_FILE_PATH  = os.getenv("EXCEL_FILE_PATH")
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")
DB_FILE_PATH = os.getenv("DB_FILE_PATH")
PARQUET_DATA_PATH = os.getenv("PARQUET_DATA_PATH")
PARQUET_CODE_PATH = os.getenv("PARQUET_CODE_PATH")
PARQUET_CODE_OUTPUT_PATH = os.getenv("PARQUET_CODE_OUTPUT_PATH")
PARQUET_OUTPUT_PATH = os.getenv("PARQUET_OUTPUT_PATH")
SEGMENT_PARQUET_PATH = os.getenv("SEGMENT_PARQUET_PATH")
FAMILY_PARQUET_PATH = os.getenv("FAMILY_PARQUET_PATH")
CLASS_PARQUET_PATH = os.getenv("CLASS_PARQUET_PATH")

# API KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# CHAT STUFF
TRAILING_PATTERN = r"(\s?>\s?)+$"
JSON_FILE_PATTERN = r'```json\s*({.*?})\s*```'
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

RETURN_FORMAT = """
    1. **Return the following structure:**
    - If you predict a Commodity title:
    ```json
    {
        - Commodity_Title: The predicted Commodity title.
        - Confidence_Score: A number between 0 and 1 representing how confident you are in this prediction.
        - Explanation: A brief explanation of why this title was chosen.
    }

    - If you predict a Class title (fallback):
    ```json
    {
        - Class_Title: The predicted Class title.
        - Confidence_Score: A number between 0 and 1 representing how confident you are in this prediction.
        - Explanation: A brief explanation of why this title was chosen.
    }
    
    2. If you **cannot predict a meaningful title**, return an error:
    ```json
    {
        "Error": "No suitable match found. Could not predict a valid UNSPSC code."
    }
"""