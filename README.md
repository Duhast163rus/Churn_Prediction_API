# 📊 Churn Prediction API

Сервис для прогнозирования оттока клиентов (Churn) на основе **FastAPI** и **MLflow**.  
Модель обучена на датасете Telco Customer Churn и зарегистрирована в MLflow Model Registry.


## 🚀 Основные возможности

- **Принимает 4 ключевых признака**:
  - `tenure` – срок контракта (месяцы)
  - `gender` – пол (`Male` / `Female` / `Мужской` / `Женский` / `м` / `ж`)
  - `age` – возраст (лет)
  - `MonthlyCharges` – ежемесячный платёж (долл.)
- **Автоматически дополняет** остальные 15 признаков значениями по умолчанию.
- **Возвращает**:
  - `churn_probability` – вероятность оттока (0..1)
  - `prediction` – `"churn"` или `"no_churn"`
  - `prediction_ru` – `"уйдет"` или `"не уйдет"` на русском
- **Гибкая загрузка модели** – из MLflow Model Registry или локальной папки.
- **Встроенные тесты** – 8 тестов покрывают валидные и ошибочные сценарии.
- **Docker-контейнер** для простого развертывания.
- **CI/CD** – автоматические тесты и сборка образа в GitHub Container Registry.

---

## 🛠 Технологии

- **Python 3.11**
- **FastAPI** – веб-фреймворк
- **Pydantic v2** – валидация данных
- **MLflow** – управление моделями
- **pytest** – тестирование
- **Docker** – контейнеризация
- **GitHub Actions** – CI/CD

---

## 📦 Установка и запуск

### Локальный запуск

1. **Клонируйте репозиторий**
   ```bash
   git clone https://github.com/your-username/churn-api.git
   cd churn-api
   ```

2. **Создайте виртуальное окружение**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. **Установите зависимости**
   ```bash
   pip install -r requirements.txt
   ```

4. **Убедитесь, что модель зарегистрирована в MLflow**
   - Папка `mlruns` должна содержать модель `churn-model` в стадии `Production`.
   - Если модели нет – обучите и зарегистрируйте её (см. раздел **Обучение модели**).

5. **Запустите сервер**
   ```bash
   python app/main.py
   ```
   Перед запуском автоматически выполнятся тесты. Если все пройдут – сервер стартует на `http://localhost:8000`.

---

### Запуск через Docker

1. **Соберите образ**
   ```bash
   docker build -t churn-api .
   ```

2. **Запустите контейнер**
   ```bash
   docker run -p 8000:8000 churn-api
   ```

Сервер будет доступен по адресу `http://localhost:8000`.

---

## 📖 Использование API

### Проверка здоровья
```http
GET /health
```
**Ответ:**
```json
{"status": "ok", "model_version": "churn-model/Production"}
```

### Предсказание
```http
POST /predict
Content-Type: application/json
```
**Тело запроса (4 обязательных поля):**
```json
{
  "tenure": 12,
  "gender": "Мужской",
  "age": 45,
  "MonthlyCharges": 75.5
}
```

**Ответ:**
```json
{
  "churn_probability": 0.42,
  "prediction": "no_churn",
  "prediction_ru": "не уйдет"
}
```

---

### Пример curl-запроса
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":12,"gender":"Мужской","age":45,"MonthlyCharges":75.5}'
```

---

## 🧪 Тестирование

Тесты находятся в папке `tests/`. Запустить их отдельно:
```bash
pytest tests/ -v
```

**8 тестов покрывают:**
- `/health` – проверка статуса
- Валидный запрос к `/predict`
- Пропуск обязательного поля → 422
- Неверный тип данных → 422
- Граничные значения (`tenure=100`, `age=65`, `MonthlyCharges=0`)
- Недопустимое значение пола → 422

---

## 🧠 Обучение и регистрация модели

Если вы хотите обучить модель заново:

1. Запустите Jupyter-ноутбук `ml-05-customer-churn.ipynb` (или ваш скрипт обучения).
2. Модель автоматически зарегистрируется в MLflow как `churn-model` в стадии `Production`.
3. Убедитесь, что в папке `mlruns` появилась запись о модели.

---

## 🐳 Docker-образ

Образ публикуется в GitHub Container Registry:

```bash
docker pull ghcr.io/your-username/churn-api:latest
docker run -p 8000:8000 ghcr.io/your-username/churn-api:latest
```

---

## 🤖 CI/CD (GitHub Actions)

При каждом push в ветку `main`:
1. Устанавливаются зависимости.
2. Запускаются тесты.
3. Собирается Docker-образ.
4. Образ публикуется в GHCR.

---

## 📁 Структура проекта

```
.
├── app/
│   ├── __init__.py
│   └── main.py              # Основной код API
├── tests/
│   └── test_api.py          # 8 тестов
├── mlruns/                  # Локальное хранилище MLflow
├── mlflow.db                # База данных MLflow (опционально)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .github/workflows/ci.yml # CI/CD пайплайн
└── README.md
```

---

## 📄 Лицензия

MIT

---

## ✨ Автор

Сергей – [GitHub](https://github.com/Duhast163rus/)

---

## 🙏 Благодарности

- Датасет: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- MLflow, FastAPI, pytest, Docker