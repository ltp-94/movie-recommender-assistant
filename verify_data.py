import polars as pl 

df = pl.read_parquet("/workspaces/movie-recommender-assistant/data/movies_with_embeddings.parquet")

print(df.head(10))