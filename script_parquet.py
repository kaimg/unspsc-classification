import polars as pl
from config import PARQUET_OUTPUT_PATH, SEGMENT_PARQUET_PATH, FAMILY_PARQUET_PATH, CLASS_PARQUET_PATH

# Load & Extract Hierarchy from UNSPSC Data
def load_and_split_unspsc():
    print("\n🔄 Loading UNSPSC data from Parquet using Polars...")

    # Read Parquet (Polars is 10x faster than Pandas)
    df = pl.read_parquet(PARQUET_OUTPUT_PATH)

    # Ensure no missing values
    df = df.fill_null("").fill_nan("")

    # Extract hierarchical levels from `Category_Path`
    df = df.with_columns(
        df["Category_Path"].str.split(" > ").alias("Hierarchy")
    )

    # Safely extract levels using `.map_elements()`
    df = df.with_columns(
        df["Hierarchy"].map_elements(lambda x: x[0] if len(x) > 0 else "").alias("Segment"),
        df["Hierarchy"].map_elements(lambda x: x[1] if len(x) > 1 else "").alias("Family"),
        df["Hierarchy"].map_elements(lambda x: x[2] if len(x) > 2 else "").alias("Class")
    ).drop("Hierarchy")  # Drop temporary hierarchy list

    print(f"Loaded & processed {df.shape[0]} UNSPSC records.\n")

    return df

# Save Unique Segment & Family Levels Separately
def save_unique_levels_as_parquet(df):
    print("\nSaving Unique Segment & Family levels as separate Parquet files...")

    # Save only unique `Segment` level
    segment_df = df.select(["Segment"]).unique().drop_nulls()
    segment_df.write_parquet(SEGMENT_PARQUET_PATH, compression="snappy")
    print(f"Unique Segment data saved at: {SEGMENT_PARQUET_PATH}")

    # Save only unique `Family` level (linked to Segment)
    family_df = df.select(["Segment", "Family"]).unique().drop_nulls()
    family_df.write_parquet(FAMILY_PARQUET_PATH, compression="snappy")
    print(f"Unique Family data saved at: {FAMILY_PARQUET_PATH}")

    # Save only unique `Family` level (linked to Segment)
    family_df = df.select(["Segment", "Family", "Class"]).unique().drop_nulls()
    family_df.write_parquet(CLASS_PARQUET_PATH, compression="snappy")
    print(f"Unique Family data saved at: {CLASS_PARQUET_PATH}")
# Run Process
df_unspsc = load_and_split_unspsc()
save_unique_levels_as_parquet(df_unspsc)
