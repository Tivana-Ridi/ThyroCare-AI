# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Thyroid_Dataset_Multiclass_FullFeatures.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ====== 1. Load dataset ======
df = pd.read_csv(DATA_PATH)

TARGET_NAME = "Thyroid_Status"

# Exclude label-like column
EXCLUDE_FEATURES = ["Thyroid_Status", "binaryClass"]
FEATURE_NAMES = [col for col in df.columns if col not in EXCLUDE_FEATURES]

print("Using features:")
for f in FEATURE_NAMES:
    print(" -", f)
print(f"\nTarget: {TARGET_NAME}")

X = df[FEATURE_NAMES]

# Encode multiclass labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[TARGET_NAME])

class_names = list(label_encoder.classes_)
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump({"classes": class_names}, f, indent=2)
print(f"\nSaved class names to {CLASS_NAMES_PATH}")
print("Classes:", class_names)

# ====== 2. Split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== 3. Pipeline ======
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, multi_class="auto"))
])

pipe.fit(X_train, y_train)

# ====== 4. Save model ======
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model_v1.pkl")
joblib.dump(pipe, MODEL_PATH)
print(f"\n✅ Saved baseline multiclass model to {MODEL_PATH}")

# ====== 5. Save feature metadata ======
feature_metadata = {
    "feature_names": FEATURE_NAMES,
    "target_name": TARGET_NAME
}
FEATURE_META_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")
with open(FEATURE_META_PATH, "w") as f:
    json.dump(feature_metadata, f, indent=2)
print(f"✅ Saved feature metadata to {FEATURE_META_PATH}")

# ====== 6. Save training stats ======
training_stats = {
    "feature_means": X_train.mean().to_dict(),
    "feature_stds": X_train.std().to_dict()
}
TRAIN_STATS_PATH = os.path.join(MODELS_DIR, "training_stats.json")
with open(TRAIN_STATS_PATH, "w") as f:
    json.dump(training_stats, f, indent=2)
print(f"✅ Saved training stats to {TRAIN_STATS_PATH}")
