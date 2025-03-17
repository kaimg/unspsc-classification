import polars as pl

# Load the Excel file starting from the correct header row
file_path = "UNSPSC English v260801.xlsx"  # Replace with your actual file path
df = pl.read_excel(file_path, read_options={"header_row": 12})

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

# Save the processed data to a CSV file
csv_file_path = "output.csv"
df.write_csv(csv_file_path)

print(f"CSV file saved as {csv_file_path}")
