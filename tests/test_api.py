import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

VALID_PAYLOAD = {
    "tenure": 1,
    "gender": "Female",
    "age": 30,
    "MonthlyCharges": 29.85
}

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "churn-model/Production"

def test_predict_valid(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "prediction_ru" in data
    assert data["prediction"] in ("churn", "no_churn")
    assert data["prediction_ru"] in ("уйдет", "не уйдет")
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_predict_missing_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["tenure"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_type(client):
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = "one"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_boundary_tenure(client):
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = 100
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_ru" in data

def test_predict_boundary_age(client):
    payload = VALID_PAYLOAD.copy()
    payload["age"] = 65
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_ru" in data

def test_predict_boundary_MonthlyCharges(client):
    payload = VALID_PAYLOAD.copy()
    payload["MonthlyCharges"] = 0.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_ru" in data

def test_predict_invalid_gender(client):
    payload = VALID_PAYLOAD.copy()
    payload["gender"] = "Unknown"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

# Новый тест: русский пол (должен пройти)
def test_predict_russian_gender(client):
    payload = VALID_PAYLOAD.copy()
    payload["gender"] = "Мужской"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_ru" in data