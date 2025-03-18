import os
import pinecone
import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import ServerlessSpec

# 🔹 File Paths
PARQUET_DATA_PATH = "/home/kaimg/Documents/p3/kamran/output.parquet"

# 🔹 Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 🔹 Initialize Pinecone
def initialize_pinecone():
    # Create an instance of the Pinecone class with your API key
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
    print("✅ Pinecone initialized.")
    return pc

# 🔹 Load & Process UNSPSC Data from Parquet
def load_unspsc_parquet(sample_size=15800):
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

# 🔹 Build and Store Data in Pinecone
def store_in_pinecone(pc, unspsc_data):
    print("\n🔄 Storing UNSPSC data in Pinecone...")

    # Create Pinecone index (if it doesn't already exist)
    index_name = "unspsc-classification"
    if index_name not in pc.list_indexes().names():
        # Define the index specification
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension should match the model output size
            metric="cosine",  # Cosine similarity is typically used for embedding vectors
            spec=spec  # Add the index specification
        )
        print(f"✅ Index '{index_name}' created.")
    else:
        print(f"✅ Index '{index_name}' already exists.")
    
    # Connect to the Pinecone index
    index = pc.Index(index_name)

    # Prepare data for insertion
    segment_titles = unspsc_data["Segment"].to_list()
    family_titles = unspsc_data["Family"].to_list()
    class_titles = unspsc_data["Class"].to_list()
    commodity_titles = unspsc_data["Commodity"].to_list()

    # Get embeddings for each title level
    segment_vectors = embedding_model.encode(segment_titles, convert_to_numpy=True).astype("float32")
    family_vectors = embedding_model.encode(family_titles, convert_to_numpy=True).astype("float32")
    class_vectors = embedding_model.encode(class_titles, convert_to_numpy=True).astype("float32")
    commodity_vectors = embedding_model.encode(commodity_titles, convert_to_numpy=True).astype("float32")

    # Prepare items for Pinecone
    segment_ids = [f"segment_{i}" for i in range(len(segment_titles))]
    family_ids = [f"family_{i}" for i in range(len(family_titles))]
    class_ids = [f"class_{i}" for i in range(len(class_titles))]
    commodity_ids = [f"commodity_{i}" for i in range(len(commodity_titles))]

    # Store the data
    index.upsert(vectors=zip(segment_ids, segment_vectors))
    index.upsert(vectors=zip(family_ids, family_vectors))
    index.upsert(vectors=zip(class_ids, class_vectors))
    index.upsert(vectors=zip(commodity_ids, commodity_vectors))

    print("✅ Data successfully stored in Pinecone.")

# 🔹 Main Function to run the process
def main():
    # Initialize Pinecone
    pc = initialize_pinecone()

    # Load and process UNSPSC data
    unspsc_data = load_unspsc_parquet()

    # Store data in Pinecone
    store_in_pinecone(pc, unspsc_data)

if __name__ == "__main__":
    main()
