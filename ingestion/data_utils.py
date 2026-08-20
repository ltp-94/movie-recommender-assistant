import polars as pl



def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans data and constructs full URLs before ingestion.
    """
    print("Pre-processing data...")

    df =  df.with_columns([
        # 1. Create Movie Link: Use IMDb if exists, else fallback to TMDB
        pl.when(pl.col("imdb_id").is_not_null())
        .then(pl.lit("https://www.imdb.com/title/") + pl.col("imdb_id") + pl.lit("/"))
        .otherwise(pl.lit("https://www.themoviedb.org/movie/") + pl.col("id").cast(pl.Utf8))
        .alias("movie_link"),

        # 2. Create Poster URL: Use TMDB CDN path, else fallback to placeholder
        pl.when(pl.col("poster_path").is_not_null())
        .then(pl.lit("https://image.tmdb.org/t/p/w500") + pl.col("poster_path"))
        .otherwise(pl.lit("https://via.placeholder.com/500x750?text=No+Poster"))
        .alias("poster_url"),

        # 3. Extract Release Year from YYYY-MM-DD string
        pl.col("release_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("release_year")
    ])

    return df