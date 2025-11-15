# Crop Prediction Web App

This repository contains a simple Flask web application that loads a Keras model (`black_pepper_model.h5`) and exposes a `/predict` endpoint to classify uploaded images (or base64 JSON images).

Contents
- `app.py` — Flask application that loads the model and provides the `/` page and `/predict` endpoint.
- `black_pepper_model.h5` — Trained Keras model (already present in repo).
- `templates/` — HTML templates (UI form at `index.html`).
- `static/` — CSS and image assets.
- `Crop_recommendation.csv` — dataset file (present but not used by `app.py`).

Requirements

Install requirements into a Python 3.8+ virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the app

```powershell
# from project root
python app.py
```

Open a browser and navigate to:

http://127.0.0.1:5000/

Using the API

The server accepts either:
- A multipart/form-data file upload field named `file` (POST to `/predict`), or
- A JSON body with a base64-encoded `image` property (data URL format: `data:image/png;base64,...`).

Example using PowerShell + curl (replace `image.jpg` with your image):

```powershell
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/predict
```

Or use the included `test_predict.py` script:

```powershell
python test_predict.py --image path\to\image.jpg
```

Notes & next steps
- The current `app.py` is an image classifier for black pepper disease classes. If you intended a crop yield prediction model (numeric regression), we can adapt the app to accept numeric features or replace the model and endpoint accordingly.
- If you're deploying to a cloud provider or Docker, I can add a `Dockerfile` and deployment instructions.
- For production use, load the model once and disable debug mode; consider adding request rate-limiting.

If you want, I can:
- Replace the UI to accept environmental features (N, P, K, temperature, etc.) and wire it to a regression model for yield prediction.
- Add tests and CI config.
