import polars as pl
from config import EXCEL_FILE_PATH, CSV_FILE_PATH
# Load the Excel file starting from the correct header row

df = pl.read_excel(EXCEL_FILE_PATH, read_options={"header_row": 12})

# Strip spaces from column names
df = df.rename({col.strip(): col.strip() for col in df.columns})

# Print column names for debugging
print("Processed Column Names:", df.columns)

# Remove "This segment includes" from Segment Definition for compactness
if "Segment Definition" in df.columns:
    df = df.with_columns(
        df["Segment Definition"]
        .str.replace(r"^This segment includes", "", literal=False)  # Remove phrase
        .str.replace(r"^\s+|\s+$", "", literal=False)  # Trim spaces manually
        .alias("Segment Definition")
    )

df.write_csv(CSV_FILE_PATH)

print(f"CSV file saved as {CSV_FILE_PATH}")
