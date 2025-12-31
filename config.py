# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_model_v1.pkl")
FEATURE_META_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.json")
TRAIN_STATS_PATH = os.path.join(BASE_DIR, "models", "training_stats.json")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

# Database
DB_PATH = os.path.join(BASE_DIR, "database", "thyroid.db")

# Model version string
MODEL_VERSION = "baseline_v1_multiclass"

# Secret key for sessions (change this in real use)
SECRET_KEY = "change-this-to-a-strong-secret-key"

