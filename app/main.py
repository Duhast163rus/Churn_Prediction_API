import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from fastapi import FastAPI, status
from pydantic import BaseModel, Field, field_validator
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
ROOT_DIR = Path(__file__).parent.parent

# ---------- Значения по умолчанию ----------
DEFAULT_VALUES = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 500.0
}

# ---------- Pydantic ----------
class ClientFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=100)
    gender: str = Field(...)
    age: int = Field(..., ge=0, le=120)
    MonthlyCharges: float = Field(..., ge=0)

    @field_validator("gender")
    def validate_gender(cls, v):
        if v in ("Male", "Female"):
            return v
        if v in ("Мужской", "мужской", "М", "м"):
            return "Male"
        if v in ("Женский", "женский", "Ж", "ж"):
            return "Female"
        raise ValueError("gender must be 'Male', 'Female', 'Мужской' or 'Женский'")

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Загрузка модели из локального хранилища MLflow...")

    # Точный путь к лучшей модели (из вашего вывода)
    model_path = ROOT_DIR / "mlruns" / "1" / "models" / "m-0cb1c91f36f4478b893888998335c93d" / "artifacts"

    if not model_path.exists():
        # Fallback: поищем любую папку с MLmodel
        mlruns_dir = ROOT_DIR / "mlruns"
        found = None
        for root, dirs, files in os.walk(mlruns_dir):
            if "MLmodel" in files:
                found = Path(root)
                break
        if found is None:
            raise FileNotFoundError(f"Не найдена модель с MLmodel в {mlruns_dir}")
        model_path = found
        logger.info(f"Найден fallback путь: {model_path}")

    model_uri = str(model_path)
    logger.info(f"Загружаем модель из {model_uri}")

    try:
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        app.state.model_version = "churn-model/best"
        logger.info("Модель успешно загружена как pyfunc")
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise

    yield
    logger.info("Выгрузка модели...")

# ---------- FastAPI ----------
app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": app.state.model_version}

@app.post("/predict")
async def predict(features: ClientFeatures):
    input_dict = features.model_dump()
    full_dict = DEFAULT_VALUES.copy()
    full_dict["tenure"] = input_dict["tenure"]
    full_dict["gender"] = input_dict["gender"]
    full_dict["MonthlyCharges"] = input_dict["MonthlyCharges"]
    full_dict["SeniorCitizen"] = 1 if input_dict["age"] >= 65 else 0
    full_dict["TotalCharges"] = input_dict["tenure"] * input_dict["MonthlyCharges"]

    input_df = pd.DataFrame([full_dict])
    model = app.state.model

    try:
        proba = model.predict_proba(input_df)[:, 1][0]
    except AttributeError:
        if hasattr(model, 'sklearn_model') and hasattr(model.sklearn_model, 'predict_proba'):
            proba = model.sklearn_model.predict_proba(input_df)[:, 1][0]
        elif hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
            proba = model._model_impl.predict_proba(input_df)[:, 1][0]
        else:
            pred = model.predict(input_df)[0]
            proba = float(pred)

    prob = float(proba)
    pred_label = "churn" if prob >= 0.5 else "no_churn"
    pred_label_ru = "уйдет" if prob >= 0.5 else "не уйдет"
    return {
        "churn_probability": prob,
        "prediction": pred_label,
        "prediction_ru": pred_label_ru
    }

# ---------- Автозапуск тестов ----------
def run_tests():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    import pytest
    print("\n=== Запуск тестов API ===\n")
    result_code = pytest.main(["tests/"])
    if result_code != 0:
        print("\n❌ Тесты не пройдены. Сервер не будет запущен.\n")
        sys.exit(1)
    else:
        print("\n✅ Все тесты успешно пройдены. Запускаем сервер...\n")

if __name__ == "__main__":
    run_tests()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)