import os
import json
import hnswlib
import polars as pl
import numpy as np
import groq
from sentence_transformers import SentenceTransformer

# 🔹 File Paths
PARQUET_DATA_PATH = "/home/kaimg/Documents/p3/kamran/output.parquet"

# 🔹 Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 🔹 Load & Process UNSPSC Data from Parquet
def load_unspsc_parquet(sample_size=1580):
    print("\n🔄 Loading UNSPSC data from Parquet using Polars...")

    df = pl.read_parquet(PARQUET_DATA_PATH).head(sample_size)
    df = df.fill_null("").fill_nan("")
    
    df = df.with_columns(
        df["Category_Path"].str.split(" > ").alias("Hierarchy")
    )
    
    # Extract each level safely using return_dtype=pl.Utf8
    df = df.with_columns(
        df["Hierarchy"].map_elements(lambda x: x[0] if len(x) > 0 else "", return_dtype=pl.Utf8).alias("Segment"),
        df["Hierarchy"].map_elements(lambda x: x[1] if len(x) > 1 else "", return_dtype=pl.Utf8).alias("Family"),
        df["Hierarchy"].map_elements(lambda x: x[2] if len(x) > 2 else "", return_dtype=pl.Utf8).alias("Class"),
        df["Hierarchy"].map_elements(lambda x: x[3] if len(x) > 3 else "", return_dtype=pl.Utf8).alias("Commodity")
    ).drop("Hierarchy")
    
    print(f"✅ Loaded & processed {df.shape[0]} UNSPSC records.\n")
    return df

UNSPSC_DATA = load_unspsc_parquet()

# 🔹 Build HNSWLIB Index (Fixing Variable Scope)
def build_hnsw_index(unspsc_data):
    print("\n🔄 Building HNSWLIB index...")

    category_levels = ["Commodity", "Class"]  # Start search from Commodity, then Class
    indexes = {}

    for level in category_levels:
        print(f"\nProcessing level: {level}")

        # ✅ Ensure columns exist in the dataset before filtering
        if level not in unspsc_data.columns:
            print(f"⚠️ Skipping level {level} because it does not exist in data.")
            continue

        select_columns = ["UNSPSC_Code", "Segment", "Family", "Class", "Commodity"]

        # ✅ Ensure every level exists, filling missing ones with ""
        for col in select_columns:
            if col not in unspsc_data.columns:
                print(f"⚠️ Column {col} not found. Adding empty values...")
                unspsc_data = unspsc_data.with_columns(pl.lit("").alias(col))

        # ✅ Fetch unique level data
        level_data = unspsc_data.filter(pl.col(level) != "").select(select_columns + [pl.col(level).alias("Level")]).unique()

        level_titles = level_data["Level"].to_list()
        unspsc_codes = level_data["UNSPSC_Code"].to_list()
        hierarchy_paths = level_data.select(["Segment", "Family", "Class", "Commodity"]).to_dicts()

        if not level_titles:
            print(f"⚠️ No entries found for level: {level}")
            continue

        print(f"Encoding {len(level_titles)} titles for level: {level}...")
        level_vectors = embedding_model.encode(level_titles, convert_to_numpy=True).astype("float32")

        num_elements, dim = level_vectors.shape
        print(f"Creating HNSWLIB index for level '{level}': {num_elements} elements, dimension {dim}")
        hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        hnsw_index.add_items(level_vectors, np.arange(num_elements))
        hnsw_index.set_ef(50)  # Tradeoff between accuracy and speed

        indexes[level] = {
            "index": hnsw_index,
            "titles": level_titles,
            "unspsc_codes": unspsc_codes,
            "hierarchy_paths": hierarchy_paths
        }
        print(f"✅ Finished building index for level: {level}")

    print(f"\n✅ HNSWLIB indexes built for {len(indexes)} levels.\n")
    return indexes

# ✅ Pass `UNSPSC_DATA` when calling `build_hnsw_index()`
HNSW_INDEXES = build_hnsw_index(UNSPSC_DATA)




# 🔹 Hierarchical HNSWLIB Search (First Try Commodity, Then Class)
def find_best_match_in_hierarchy(query_text, confidence_threshold=0.8):
    print(f"\n🔍 Searching HNSWLIB for: {query_text}")

    for level in ["Commodity", "Class"]:  # Start at Commodity, fallback to Class
        if level not in HNSW_INDEXES:
            continue  # Skip missing levels

        print(f"🔎 Searching at level: {level}")
        index_data = HNSW_INDEXES[level]
        index, titles, unspsc_codes, hierarchy_paths = (
            index_data["index"], 
            index_data["titles"], 
            index_data["unspsc_codes"], 
            index_data["hierarchy_paths"]
        )

        query_vector = embedding_model.encode([query_text], convert_to_numpy=True).astype("float32")

        labels, distances = index.knn_query(query_vector, k=1)
        matched_index = labels[0][0]
        similarity_score = float(1 - distances[0][0])  # ✅ Convert to Python `float`

        if matched_index >= 0 and similarity_score >= confidence_threshold:
            matched_text = titles[matched_index]
            best_unspsc_code = unspsc_codes[matched_index]
            hierarchy_path = hierarchy_paths[matched_index]  # Get full hierarchy

            print(f"✅ Matched {level}: {matched_text} (Confidence: {similarity_score:.2f})")

            return {
                "UNSPSC_Code": best_unspsc_code,
                "Matched_Level": level,
                "Matched_Text": matched_text,
                "Confidence_Score": round(float(similarity_score), 4),
                "Hierarchy_Path": hierarchy_path
            }

    print("⚠️ No confident match found at Commodity or Class level.\n")
    return None

# 🔹 Main Function: HNSWLIB-Based UNSPSC Retrieval
def get_unspsc_code(query_text):
    print(f"\n🔍 Searching for UNSPSC Code...")

    dataset_match = find_best_match_in_hierarchy(query_text)

    if dataset_match:
        print("\n✅ HNSWLIB found a high-confidence UNSPSC match!")
        return json.dumps(dataset_match, indent=4)  # ✅ JSON serializable

    print("\n🤖 HNSWLIB is unsure. No UNSPSC Code found.")
    return json.dumps({"Error": "No match found in HNSWLIB."}, indent=4)

# 🔹 Main Loop
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter product description: ")
        if user_input.lower() in ["exit", "quit"]:
            print("🚪 Exiting...")
            break
        result = get_unspsc_code(user_input)
        print(result)
