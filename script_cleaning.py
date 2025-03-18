import polars as pl
from config import PARQUET_DATA_PATH, PARQUET_OUTPUT_PATH, TRAILING_PATTERN

# Function to Remove Trailing " > "
def clean_trailing_arrows():
    print("\nLoading PARQUET and cleaning trailing '>' symbols...")

    # Load PARQUET
    df = pl.read_parquet(PARQUET_DATA_PATH)

    # Check if "Category_Path" exists
    if "Category_Path" not in df.columns:
        print("Error: 'Category_Path' column not found in PARQUET!")
        return

    # Remove trailing " > " symbols
    df = df.with_columns(
        pl.col("Category_Path").str.replace(TRAILING_PATTERN, "")  # Remove trailing " > "
    )

    print(f"Fixed {df.height} rows with trailing '>'.")

    # Save cleaned data to new PARQUET
    df.write_parquet(PARQUET_OUTPUT_PATH)
    print(f"Cleaned data saved to: {PARQUET_OUTPUT_PATH}")

# Run Cleaning Function
clean_trailing_arrows()
