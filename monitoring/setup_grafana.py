import os
import json
import requests
import time

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin") # Або ваші змінні з .env

def setup_grafana():
    # 1. Чекаємо на Grafana
    while True:
        try:
            if requests.get(f"{GRAFANA_URL}/api/health").status_code == 200: break
        except: pass
        print("⏳ Waiting for Grafana...")
        time.sleep(3)

    # 2. Створюємо Datasource (PostgreSQL)
    ds_payload = {
        "name": "PostgreSQL",
        "type": "postgres",
        "url": "postgres:5432", # Назва сервісу в docker-compose
        "access": "proxy",
        "user": "admin",
        "database": "movie_db",
        "jsonData": {"sslmode": "disable", "postgresVersion": 15},
        "secureJsonData": {"password": "password"}
    }
    requests.delete(f"{GRAFANA_URL}/api/datasources/name/PostgreSQL", auth=AUTH)
    resp = requests.post(f"{GRAFANA_URL}/api/datasources", json=ds_payload, auth=AUTH)
    ds_uid = resp.json().get("uid", "PostgreSQL")
    print(f"✅ Datasource configured (UID: {ds_uid})")

    # 3. Завантажуємо Dashboard
    with open("monitoring/dashboard.json", "r") as f:
        db_json = json.load(f)

    # Примусово прив'язуємо панелі до нашого UID
    for panel in db_json.get("panels", []):
        panel["datasource"] = {"type": "postgres", "uid": ds_uid}
        if "targets" in panel:
            for t in panel["targets"]:
                t["datasource"] = {"type": "postgres", "uid": ds_uid}

    payload = {"dashboard": db_json, "overwrite": True}
    res = requests.post(f"{GRAFANA_URL}/api/dashboards/db", json=payload, auth=AUTH)
    
    if res.status_code == 200:
        print("🚀 Dashboard Ready! View it at: http://localhost:3000/d/movie-assistant")
    else:
        print(f"❌ Error: {res.text}")

if __name__ == "__main__":
    setup_grafana()