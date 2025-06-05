import polars as pl
from config import CSV_FILE_PATH, PARQUET_DATA_PATH, PARQUET_CODE_PATH

# Load & Optimize UNSPSC Data
def load_and_optimize_unspsc(sample_size=158000):
    print("\nLoading & optimizing UNSPSC data using Polars...")

    # Read CSV with forced string type (Utf8) to avoid type mismatch errors
    df = pl.read_csv(
        CSV_FILE_PATH,
        infer_schema_length=10000,  # Ensures sufficient rows are checked
        ignore_errors=True  # Prevents crashes on bad data
    ).head(sample_size)

    # Ensure required columns exist (Avoid ColumnNotFoundError)
    required_columns = ["Key", "Segment Title", "Family Title", "Class Title", "Commodity Title"]
    need_columns = ["Key", "Segment", "Family", "Class", "Commodity"]
    for col in need_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in CSV. Adding empty values.")
            df = df.with_columns(pl.lit("").alias(col))  # Fill missing columns with ""

    # Merge category codes into a single column
    df = df.with_columns(
        (df["Segment"].fill_null("").cast(str) + " > " + 
         df["Family"].fill_null("").cast(str) + " > " + 
         df["Class"].fill_null("").cast(str) + " > " + 
         df["Commodity"].fill_null("").cast(str)).alias("UNSPSC_Path")
    )

    # Merge category titles into a single column
    df = df.with_columns(
        (df["Segment Title"].fill_null("").cast(str) + " > " + 
         df["Family Title"].fill_null("").cast(str) + " > " + 
         df["Class Title"].fill_null("").cast(str) + " > " + 
         df["Commodity Title"].fill_null("").cast(str)).alias("Category_Path")
    )

    # Keep only necessary columns
    df = df.select(["Key", "Category_Path"]).rename({"Key": "UNSPSC_Code"})

    print(f"Optimized Data: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

# Load & Optimize UNSPSC Data
def load_and_optimize_unspsc_code(sample_size=158000):
    print("\nLoading & optimizing UNSPSC data using Polars...")

    # Read CSV with forced string type (Utf8) to avoid type mismatch errors
    df = pl.read_csv(
        CSV_FILE_PATH
    ).head(sample_size)

    # Ensure required columns exist (Avoid ColumnNotFoundError)
    need_columns = ["Key", "Segment", "Family", "Class", "Commodity"]
    for col in need_columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in CSV. Adding empty values.")
            df = df.with_columns(pl.lit("").alias(col))

    # Merge category codes into a single column
    df = df.with_columns(
        (df["Segment"].fill_null("").cast(str) + " > " + 
         df["Family"].fill_null("").cast(str) + " > " + 
         df["Class"].fill_null("").cast(str) + " > " + 
         df["Commodity"].fill_null("").cast(str)).alias("UNSPSC_Path")
    )
    
    # Keep only necessary columns
    df = df.select(["Key", "UNSPSC_Path"]).rename({"Key": "UNSPSC_Code"})
    print(df)

    print(f"Optimized Data: {df.shape[0]} rows, {df.shape[1]} columns\n")
    df.write_parquet(PARQUET_CODE_PATH, compression="snappy")
    return df

# Save to Parquet (Faster than CSV)
def save_to_parquet(df):
    print("\nSaving optimized data to Parquet (Polars)...")
    df.write_parquet(PARQUET_DATA_PATH, compression="snappy")
    print(f"Data saved as Parquet at: {PARQUET_DATA_PATH}\n")

# Save to Parquet (Faster than CSV)
def save_to_parquet_code(df):
    print("\nSaving optimized data to Parquet (Polars)...")
    df.write_csv(PARQUET_CODE_PATH, compression="snappy")
    print(f"Data saved as Parquet at: {PARQUET_CODE_PATH}\n")

# Run Optimization
#df_optimized = load_and_optimize_unspsc()
#save_to_parquet(df_optimized)

df_code = load_and_optimize_unspsc_code()
# save_to_parquet_code(df_code)
