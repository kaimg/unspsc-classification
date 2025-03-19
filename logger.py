import logging
from pathlib import Path

# 🔹 Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)

# 🔹 Configure logging
log_file = logs_dir / "app.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)
# Create a logger
logger = logging.getLogger(__name__)

# def fetch_json_response(response):
#     try:
#         logger.info("Attempting to parse JSON from response...")
#         return json.dumps(response)
#     except ValueError as err:
#         logger.error(f"Failed to parse JSON: {err}")
#         return json.dumps({"Error": f"No JSON found in the response {err=}, {type(err)=}"}, indent=4)
# 
# def main():
#     logger.info("Starting the application...")
#     response = "invalid_json"  # Simulate a response that is not valid JSON
#     result = fetch_json_response(response)
#     logger.info(f"Result: {result}")
#     logger.info("Application finished.")
# 
# if __name__ == "__main__":
#     main()