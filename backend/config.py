import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline.pkl")
    METADATA_PATH = os.getenv("METADATA_PATH", "models/model_metadata.json")
    HISTORY_FILE = os.getenv("HISTORY_FILE", "data/prediction_history.csv")
    BACKGROUND_DATA_PATH = os.getenv("BACKGROUND_DATA_PATH", "models/background_data.csv")
    PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

config = Config()
