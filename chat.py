import os
import re
import json
import hnswlib
import polars as pl
import numpy as np
import groq
from sentence_transformers import SentenceTransformer

# 🔹 File Paths
PARQUET_DATA_PATH = "/home/kaimg/Documents/p3/kamran/output.parquet"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure to set this environment variable

# 🔹 Load Sentence Transformer Model for Embeddings
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 🔹 Initialize Groq Client for LLM (Mistral)
client = groq.Client(api_key=GROQ_API_KEY)

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

UNSPSC_DATA = load_unspsc_parquet()

# 🔹 Build HNSWLIB Index
def build_hnsw_index(unspsc_data):
    print("\n🔄 Building HNSWLIB index...")
    category_levels = ["Commodity", "Class"]
    indexes = {}

    for level in category_levels:
        print(f"\nProcessing level: {level}")
        level_data = unspsc_data.filter(pl.col(level) != "").select(["UNSPSC_Code", "Segment", "Family", "Class", "Commodity"]).unique()
        level_titles = level_data[level].to_list()
        unspsc_codes = level_data["UNSPSC_Code"].to_list()
        hierarchy_paths = level_data.select(["Segment", "Family", "Class", "Commodity"]).to_dicts()

        level_vectors = embedding_model.encode(level_titles, convert_to_numpy=True).astype("float32")
        num_elements, dim = level_vectors.shape
        hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        hnsw_index.add_items(level_vectors, np.arange(num_elements))
        hnsw_index.set_ef(50)

        indexes[level] = {
            "index": hnsw_index,
            "titles": level_titles,
            "unspsc_codes": unspsc_codes,
            "hierarchy_paths": hierarchy_paths
        }
        print(f"✅ Finished building index for level: {level}")

    return indexes

HNSW_INDEXES = build_hnsw_index(UNSPSC_DATA)

# 🔹 Search Logic using LLM and HNSW
def find_best_match_in_hierarchy(query_text, confidence_threshold=0.8):
    print(f"\n🔍 Searching for: {query_text}")

    for level in ["Commodity", "Class"]:
        index_data = HNSW_INDEXES[level]
        index, titles, unspsc_codes, hierarchy_paths = index_data["index"], index_data["titles"], index_data["unspsc_codes"], index_data["hierarchy_paths"]

        query_vector = embedding_model.encode([query_text], convert_to_numpy=True).astype("float32")
        labels, distances = index.knn_query(query_vector, k=1)
        matched_index = labels[0][0]
        similarity_score = float(1 - distances[0][0])

        if matched_index >= 0 and similarity_score >= confidence_threshold:
            matched_text = titles[matched_index]
            best_unspsc_code = unspsc_codes[matched_index]
            hierarchy_path = hierarchy_paths[matched_index]

            print(f"✅ Matched {level}: {matched_text} (Confidence: {similarity_score:.2f})")

            return {
                "UNSPSC_Code": best_unspsc_code,
                "Matched_Level": level,
                "Matched_Text": matched_text,
                "Confidence_Score": round(float(similarity_score), 4),
                "Hierarchy_Path": hierarchy_path
            }
    return None

# 🔹 Main Search Loop
def get_unspsc_code(query_text):
    print(f"\n🔍 Searching for UNSPSC Code...")
    SYSTEM_PROMPT = """
        You are an AI trained to enrich product descriptions and predict the most appropriate UNSPSC commodity or class title based on the given description. You will work with a hierarchical structure that includes:
        1. Segment
        2. Family
        3. Class
        4. Commodity

        Your tasks are as follows:
        1. **Check if the product description is sufficient for finding a match.**  
        If the description is clear enough, **enrich it with additional details** (e.g., synonyms, abbreviations, etc.) and **predict the most appropriate Commodity title**.
        
        2. **If the description is not sufficient** or if you cannot confidently predict the Commodity title, return the original description, and **predict the best Class title** instead.

        3. **Return the following structure:**
        - If you predict a Commodity title:
        ```json
        {
            - Commodity_Title: The predicted Commodity title.
            - Confidence_Score: A number between 0 and 1 representing how confident you are in this prediction.
            - Explanation: A brief explanation of why this title was chosen.
        }

        - If you predict a Class title (fallback):
        ```json
        {
            - Class_Title: The predicted Class title.
            - Confidence_Score: A number between 0 and 1 representing how confident you are in this prediction.
            - Explanation: A brief explanation of why this title was chosen.
        }
        
        4. If you **cannot predict a meaningful title**, return an error:
        ```json
        {
            "Error": "No suitable match found. Could not predict a valid UNSPSC code."
        }

        5. Remember: You should ALWAYS in the end of answer returnING in json format.
    """
    # Step 1: Ask LLM for enriched prediction
    llm_prompt = f"Process the product description and predict the most relevant Commodity or Class title for: {query_text}"
    response = client.chat.completions.create(
        model="mistral-saba-24b",  # Use Mistral model from Groq API
        messages=[{"role": "system", "content": "<SYS> " + SYSTEM_PROMPT}, {"role": "user", "content": llm_prompt}]
    )
    # Extract predictions and confidence scores
    llm_response = response.choices[0].message.content
    print(f"\nLLM Response: {llm_response}")

    # Extract the title from LLM response (either Commodity or Class)
    # You will need to parse this response as per your output format

    pattern = r'```json\s*({.*?})\s*```'
    match = re.search(pattern, llm_response, re.DOTALL)

    try:
        if match:
            llm_json = json.loads(match.group(1))
            print(llm_json)
            # Check if either 'Commodity_Title' or 'Class_Title' exists in the JSON
            if 'Commodity_Title' in llm_json:
                title = llm_json['Commodity_Title']
                print(f"Commodity Title: {title}")
            elif 'Class_Title' in llm_json:
                title = llm_json['Class_Title']
                print(f"Class Title: {title}")
            else:
                return json.dumps({"Error": "No suitable match found. Could not predict a valid UNSPSC code."}, indent=4)
    except Exception as err:
        print(f"No JSON found in the response {err=}, {type(err)=}")

    # If LLM provides a valid prediction, proceed with the HNSW search
    dataset_match = find_best_match_in_hierarchy(title)

    if dataset_match:
        print(json.dumps(dataset_match, indent=4))
        return json.dumps(dataset_match, indent=4)

    print("\n🤖 Unable to find a suitable match.")
    return json.dumps({"Error": "No match found."}, indent=4)

# 🔹 Main Function Loop
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nEnter product description: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🚪 Exiting...")
                break
            result = get_unspsc_code(user_input)
            print(result)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nSearch interrupted. Press Ctrl+C again to exit or continue searching.")
            continue  # Restart the loop to allow searching again
