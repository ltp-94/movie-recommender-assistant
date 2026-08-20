import os
from dotenv import load_dotenv

# 1. Load env vars first!
load_dotenv()

# 2. Set this to skip the check in db.py if needed
os.environ['RUN_TIMEZONE_CHECK'] = '0'

# 3. Now import your db functions
from monitoring.db import init_db

if __name__ == "__main__":
    print("🚀 Initializing database...")
    try:
        init_db()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")