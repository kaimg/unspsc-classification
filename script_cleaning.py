import polars as pl
import re

# 🔹 File Paths
CSV_INPUT_PATH = "/home/kaimg/Documents/p3/kamran/output-from.csv"
CSV_OUTPUT_PATH = "/home/kaimg/Documents/p3/kamran/output-from-fixed.csv"

# 🔹 Function to Remove Trailing " > "
def clean_trailing_arrows():
    print("\n🔄 Loading CSV and cleaning trailing '>' symbols...")

    # ✅ Load CSV
    df = pl.read_csv(CSV_INPUT_PATH, infer_schema_length=10000, ignore_errors=True)

    # ✅ Check if "Category_Path" exists
    if "Category_Path" not in df.columns:
        print("❌ Error: 'Category_Path' column not found in CSV!")
        return

    # ✅ Remove trailing " > " symbols
    df = df.with_columns(
        df["Category_Path"].str.replace_all(r"(\s?>\s?)+$", "")  # ✅ Remove trailing " > "
    )

    print(f"✅ Fixed {df.shape[0]} rows with trailing '>'.")

    # ✅ Save cleaned data to new CSV
    df.write_csv(CSV_OUTPUT_PATH)
    print(f"✅ Cleaned data saved to: {CSV_OUTPUT_PATH}")

# 🔹 Run Cleaning Function
clean_trailing_arrows()
