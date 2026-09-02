import os
import gdown
import polars as pl


# ============================================================
# DOWNLOAD DATASET
# ============================================================
DATA_DIR = "/workspaces/movie-recommender-assistant/data"
os.makedirs(DATA_DIR, exist_ok=True)
FILE_ID = "1c0J8VrBW9vHfkYyOBVJJsrq54b3swGPU"
OUTPUT = f"{DATA_DIR}/movies_with_embeddings.parquet"

if not os.path.exists(OUTPUT):
   print("Downloading dataset...")
   gdown.download(
       f"https://drive.google.com/uc?id={FILE_ID}",
       OUTPUT,
       quiet=False,
   )
