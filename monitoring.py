import pandas as pd
import sqlite3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import datetime

# Подключаемся к базе с логами предсказаний (создадим позже)
conn = sqlite3.connect("predictions.db")

# Загружаем эталонные данные (первые 1000 строк из train)
# (здесь нужно загрузить X_train из вашего датасета)
reference = pd.read_csv("data.csv").sample(1000)  # пример

# Загружаем новые данные (например, за последний час)
current = pd.read_sql_query("SELECT * FROM predictions WHERE timestamp > datetime('now', '-1 day')", conn)

if len(current) > 0:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("drift_report.html")
    print("Отчёт сохранён в drift_report.html")
else:
    print("Нет новых данных для мониторинга")