import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import shap
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("🔮 Предсказание оттока клиента")

# ---------- Статичные значения по умолчанию ----------
DEFAULT_VALUES = {
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

RUSSIAN_NAMES = {
    "gender": "Пол",
    "SeniorCitizen": "Пенсионер (0/1)",
    "Partner": "Партнёр",
    "Dependents": "Иждивенцы",
    "tenure": "Стаж (месяцы)",
    "PhoneService": "Телефонная линия",
    "MultipleLines": "Несколько линий",
    "InternetService": "Интернет-услуга",
    "OnlineSecurity": "Онлайн-безопасность",
    "OnlineBackup": "Онлайн-бэкап",
    "DeviceProtection": "Защита устройств",
    "TechSupport": "Техподдержка",
    "StreamingTV": "Стриминг ТВ",
    "StreamingMovies": "Стриминг фильмов",
    "PaperlessBilling": "Электронный счёт",
    "PaymentMethod": "Способ оплаты",
    "MonthlyCharges": "Ежемесячный платёж ($)",
    "TotalCharges": "Общая сумма ($)"
}

with st.expander("📋 Данные по умолчанию для недостающих признаков"):
    df_display = pd.DataFrame([
        {"Признак": RUSSIAN_NAMES.get(key, key), "Значение": value}
        for key, value in DEFAULT_VALUES.items()
    ])
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.caption("Эти значения используются для признаков, которые вы не передаёте в запросе.")


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

@st.cache_resource
def load_model():
    """Загружает модель из локальной папки mlruns, ища папку с MLmodel."""
    mlruns_dir = Path("mlruns")
    model_path = None
    for root, dirs, files in os.walk(mlruns_dir):
        if "MLmodel" in files:
            model_path = Path(root)
            break
    if model_path is None:
        st.error("Модель не найдена в папке mlruns")
        return None
    mlflow.set_tracking_uri("file:./mlruns")
    return mlflow.pyfunc.load_model(str(model_path))

@st.cache_resource
def get_explainer(_model):
    """Создаёт SHAP explainer для модели."""
    if _model is None:
        return None
    inner_model = None
    if hasattr(_model, 'sklearn_model'):
        inner_model = _model.sklearn_model
    elif hasattr(_model, '_model_impl'):
        inner_model = _model._model_impl
    else:
        inner_model = _model

    if not hasattr(inner_model, 'predict_proba'):
        return None

    try:
        # Создаём фоновый датасет
        if hasattr(inner_model, 'feature_names_in_'):
            feature_names = inner_model.feature_names_in_
        else:
            feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                             'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                             'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                             'MonthlyCharges', 'TotalCharges']
        np.random.seed(42)
        n = 100
        data = {
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'Dependents': np.random.choice(['Yes', 'No'], n),
            'tenure': np.random.randint(0, 100, n),
            'PhoneService': np.random.choice(['Yes', 'No'], n),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n),
            'MonthlyCharges': np.random.uniform(20, 120, n),
            'TotalCharges': np.random.uniform(100, 5000, n)
        }
        background_df = pd.DataFrame(data)
        if hasattr(inner_model, 'feature_names_in_'):
            background_df = background_df[inner_model.feature_names_in_]
        explainer = shap.Explainer(inner_model.predict_proba, background_df)
        return explainer
    except Exception as e:
        st.warning(f"Не удалось создать SHAP explainer: {e}")
        return None

model = load_model()
explainer = get_explainer(model)

st.subheader("Введите данные клиента")
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Стаж (месяцы)", min_value=0, max_value=100, value=12)
    age = st.number_input("Возраст", min_value=18, max_value=100, value=45)
with col2:
    gender = st.selectbox("Пол", ["Male", "Female", "Мужской", "Женский"])
    monthly_charges = st.number_input("Ежемесячный платёж ($)", min_value=0.0, value=75.5)

contract = st.selectbox(
    "Тип контракта",
    ["Month-to-month", "One year", "Two year"],
    index=0
)

if st.button("Рассчитать вероятность"):
    payload = {
        "tenure": tenure,
        "gender": gender,
        "age": age,
        "MonthlyCharges": monthly_charges,
        "contract": contract
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            pred = result["prediction_ru"]
            st.success(f"**Вероятность оттока:** {prob:.2f}")
            st.success(f"**Прогноз:** {pred}")

            if explainer is not None:
                full_features = {
                    "gender": "Male" if gender in ["Male", "Мужской", "М"] else "Female",
                    "SeniorCitizen": 1 if age >= 65 else 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": tenure,
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
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": tenure * monthly_charges
                }
                input_df = pd.DataFrame([full_features])
                if hasattr(explainer, 'model') and hasattr(explainer.model, 'feature_names_in_'):
                    input_df = input_df[explainer.model.feature_names_in_]
                shap_values = explainer(input_df)
                st.subheader("📈 SHAP-объяснение")
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(shap_values[0], show=False)
                st.pyplot(fig)
            else:
                st.info("SHAP-анализ недоступен для этой модели.")
        else:
            st.error(f"Ошибка API: {response.status_code}")
    except Exception as e:
        st.error(f"Не удалось подключиться к API: {e}")