import polars as pl 
CSV_FILE_PATH = "output.csv"
PARQUET_FILE_PATH = "output-from-fixed.parquet"
OUTPUT_CSV_PATH = "test.csv"



# Read master CSV
unspsc = pl.read_csv(CSV_FILE_PATH)
unspsc = unspsc.rename({col: col.strip() for col in unspsc.columns})

# Read input parquet
parquet_data = pl.read_parquet(PARQUET_FILE_PATH)

# Add hierarchy level
parquet_data = parquet_data.with_columns([
    pl.col("Category_Path").map_elements(lambda x: x.count(">"), return_dtype=pl.Int64).alias("Level")
])

# Map levels to UNSPSC columns
level_to_column = {
    0: "Segment",
    1: "Family",
    2: "Class",
    3: "Commodity"
}

# Process each level
results = []
for level, hierarchy_col in level_to_column.items():
    filtered = parquet_data.filter(pl.col("Level") == level)

    joined = filtered.join(
        unspsc.select([hierarchy_col, "Key"]),
        left_on="UNSPSC_Code",
        right_on="Key",
        how="left"
    ).with_columns([
        pl.col(hierarchy_col).alias("UNSPSC_Code_Full"),
        pl.col(hierarchy_col).alias("UNSPSC_Level_Code")
    ]).select([
        "UNSPSC_Code", "Category_Path", "Level", "UNSPSC_Code_Full", "UNSPSC_Level_Code"
    ])

    results.append(joined)

# Concatenate all cleaned pieces
final_df = pl.concat(results)

# Save output
final_df.write_csv(OUTPUT_CSV_PATH)
print(f"✅ Output written to: {OUTPUT_CSV_PATH}")



