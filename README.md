# ThyroCare-AI

ThyroCare-AI is a Flask web app for multi-class thyroid status prediction with
user accounts, prediction history, and optional SHAP-based explanations.

## Features
- Multi-class prediction using an ensemble (logistic regression + random forest)
  with calibrated probabilities
- User signup/login with SQLite persistence
- Prediction history and monitoring summary
- Lightweight feature-importance explanation
- Optional SHAP waterfall visualizations

## Tech Stack
- Python, Flask, scikit-learn, NumPy, pandas, SQLite
- SHAP + matplotlib (optional for explainability)

## Project Structure
- `app.py`: Flask app, API routes, prediction logic
- `config.py`: model, database, and secret key config
- `database/init_db.py`: creates SQLite tables
- `train_model.py`: trains and saves model artifacts
- `models/`: saved model and metadata files
- `templates/`, `static/`: frontend UI

## Setup
Prereqs: Python 3.10+ recommended

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python database/init_db.py
python app.py
```

App runs at `http://localhost:5000`.

## Model Artifacts
`app.py` loads:
- `models/baseline_model_v1.pkl`
- `models/feature_metadata.json`
- `models/training_stats.json`
- `models/class_names.json`

The loaded model is expected to be a dict with keys `scaler`, `lr_cal`, and
`rf_cal` and matching class ordering.

## Training (Optional)
If you want to retrain:

```bash
python train_model.py
```

`train_model.py` expects `data/Thyroid_Dataset_Multiclass_FullFeatures.csv` and
writes new artifacts to `models/`.

## SHAP Explanations (Optional)
If SHAP is available and `models/shap_background.npy` exists, the app can
generate SHAP waterfall plots via `/api/shap-waterfall`.

## Notes
- Change `SECRET_KEY` in `config.py` for production use.
- SQLite database file: `database/thyroid.db`
