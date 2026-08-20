import os
import json
import requests
import time
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")

# Postgres settings (must match your docker-compose service name)
PG_HOST = os.getenv("POSTGRES_HOST", "postgres") 
PG_DB = os.getenv("POSTGRES_DB", "movie_db")
PG_USER = os.getenv("POSTGRES_USER", "admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")

# Auth tuple for requests
AUTH = (GRAFANA_USER, GRAFANA_PASSWORD)

def wait_for_grafana():
    """Wait until Grafana API is actually ready."""
    print("⏳ Waiting for Grafana to start...")
    for i in range(10):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health")
            if response.status_code == 200:
                print("✅ Grafana is UP!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    return False

def setup_datasource():
    """Create the PostgreSQL connection in Grafana."""
    print("🔗 Configuring PostgreSQL Datasource...")
    
    payload = {
        "name": "PostgreSQL",
        "type": "postgres",
        "url": f"{PG_HOST}:{PG_PORT}",
        "access": "proxy",
        "user": PG_USER,
        "database": PG_DB,
        "basicAuth": False,
        "isDefault": True,
        "jsonData": {
            "sslmode": "disable",
            "postgresVersion": 1300
        },
        "secureJsonData": {
            "password": PG_PASSWORD
        }
    }

    # Try to delete if it exists to avoid conflicts
    requests.delete(f"{GRAFANA_URL}/api/datasources/name/PostgreSQL", auth=AUTH)

    response = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        json=payload,
        auth=AUTH
    )

    if response.status_code in [200, 201]:
        print("✅ Datasource created successfully.")
        return response.json().get("uid")
    else:
        print(f"❌ Failed to create datasource: {response.text}")
        return None

def setup_dashboard(datasource_uid):
    """Upload the dashboard.json file to Grafana."""
    print("📊 Uploading Movie Assistant Dashboard...")
    
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.json")
    
    try:
        with open(dashboard_path, "r") as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {dashboard_path} not found.")
        return

    # Force the dashboard to use the new datasource UID
    for panel in dashboard_json.get("panels", []):
        if "datasource" in panel and isinstance(panel["datasource"], dict):
            panel["datasource"]["uid"] = datasource_uid

    payload = {
        "dashboard": dashboard_json,
        "overwrite": True
    }

    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        json=payload,
        auth=AUTH
    )

    if response.status_code == 200:
        print("✅ Dashboard created successfully!")
    else:
        print(f"❌ Failed to create dashboard: {response.text}")

def main():
    if wait_for_grafana():
        ds_uid = setup_datasource()
        if ds_uid:
            setup_dashboard(ds_uid)

if __name__ == "__main__":
    main()