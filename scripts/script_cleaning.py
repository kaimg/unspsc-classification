import polars as pl
from config import PARQUET_DATA_PATH, PARQUET_OUTPUT_PATH, PARQUET_CODE_PATH, PARQUET_CODE_OUTPUT_PATH, TRAILING_PATTERN

# Function to Remove Trailing " > "
def clean_trailing_arrows(infile, outfile, column_name="Category_Path"):
    print("\nLoading PARQUET and cleaning trailing '>' symbols...")

    # Load PARQUET
    df = pl.read_parquet(infile)
    print(infile)
    # Check if column_name exists
    if column_name not in df.columns:
        print(f"Error: {column_name} column not found in PARQUET!")
        return

    # Remove trailing " > " symbols
    pattern = TRAILING_PATTERN
    df = df.with_columns(
        pl.col(column_name).str.replace(pattern, "")  # Remove trailing " > "
    )

    print(f"Fixed {df.height} rows with trailing '>'.")
    print(df)
    # Save cleaned data to new PARQUET

    print(outfile)
    print(f"Cleaned data saved to: {outfile}")
    df.write_parquet(outfile)
    print(f"Cleaned data saved to: {outfile}")

# Run Cleaning Function
#clean_trailing_arrows(PARQUET_DATA_PATH, PARQUET_OUTPUT_PATH)
clean_trailing_arrows(PARQUET_CODE_PATH, PARQUET_CODE_OUTPUT_PATH, "UNSPSC_Path")

