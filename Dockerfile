FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зеркало PyPI (быстрее в РФ) и увеличиваем таймаут
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY app/ ./app/
COPY tests/ ./tests/
COPY mlruns/ ./mlruns/
COPY dashboard.py .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]