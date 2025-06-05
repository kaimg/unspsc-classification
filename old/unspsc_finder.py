# DATA LOADING
import pandas as pd

file_path = "./data/unspsc_data.csv"
data = pd.read_csv(file_path)
data["Segment"] = data["Segment"].astype("Int64").astype(str)
data["Family"] = data["Family"].astype("Int64").astype(str)
data["Class"] = data["Class"].astype("Int64").astype(str)
data["Commodity"] = data["Commodity"].astype("Int64").astype(str)
data.rename(
    columns={c: c.lower().replace(" ", "_") for c in data.columns}, inplace=True
)

print(data.sample().to_xml(index=False))
# <?xml version='1.0' encoding='utf-8'?>
# <data>
#   <row>
#     <segment>41000000</segment>
#     <segment_title>Laboratory and Measuring and Observing and Testing Equipment</segment_title>
#     <segment_definition> the machines, equipment and tools used in laboratories, as well as measuring, observing and testing equipment</segment_definition>
#     <family>41140000</family>
#     <family_title>Clinical chemistry testing systems, components, and supplies</family_title>
#     <family_definition>In this classification these entries connote systems, components and supplies used in clinical chemistry based medical procedures.</family_definition>
#     <class>41141800</class>
#     <class_title>Clinical chemistry substrates</class_title>
#     <class_definition/>
#     <commodity>41141822</commodity>
#     <commodity_title>Non esterified fatty acids</commodity_title>
#     <commodity_definition>In this classification, this entry connotes a substrate made of Non esterified fatty acids that is used in clinical chemistry testing.</commodity_definition>
#   </row>
# </data>


segment_count = data[pd.isna(data["family"])].shape[0]
family_count = data[pd.isna(data["class"]) & ~pd.isna(data["family"])].shape[0]
class_count = data[pd.isna(data["commodity"]) & ~pd.isna(data["class"])].shape[0]
commodity_count = data[~pd.isna(data["commodity"])].shape[0]
sum([segment_count, family_count, class_count, commodity_count])
print(
    f"Segment: {segment_count}, Family: {family_count}, Class: {class_count}, Commodity: {commodity_count}"
)
# Segment: 58, Family: 559, Class: 7997, Commodity: 149834

# END OF DATA LOADING
20*20*20*20

from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceScore(BaseModel):
    """Confidence score for a UNSPSC match."""

    score: float = Field(
        ..., ge=0, le=1, description="Confidence score between 0 and 1"
    )
    explanation: str = Field(..., description="Explanation for the confidence score")


class UNSPSCMatch(BaseModel):
    """Hierarchical UNSPSC match with confidence scores."""

    segment: Optional[str] = Field(None, description="8-digit UNSPSC segment code")
    segment_title: Optional[str] = Field(None, description="Title of the segment")
    segment_confidence: Optional[ConfidenceScore] = None

    family: Optional[str] = Field(None, description="8-digit UNSPSC family code")
    family_title: Optional[str] = Field(None, description="Title of the family")
    family_confidence: Optional[ConfidenceScore] = None

    class_code: Optional[str] = Field(None, description="8-digit UNSPSC class code")
    class_title: Optional[str] = Field(None, description="Title of the class")
    class_confidence: Optional[ConfidenceScore] = None

    commodity: Optional[str] = Field(None, description="8-digit UNSPSC commodity code")
    commodity_title: Optional[str] = Field(None, description="Title of the commodity")
    commodity_confidence: Optional[ConfidenceScore] = None

    cleaned_description: str = Field(..., description="Cleansed input description")
    overall_explanation: str = Field(
        ..., description="Overall explanation of the classification"
    )


from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# Create documents for each level of hierarchy
def create_hierarchy_documents(df):
    documents = []

    # Segment level
    for _, row in df[pd.isna(df["family"])].iterrows():
        doc = Document(
            text=f"SEGMENT: {row['segment_title']}\nDEFINITION: {row['segment_definition']}",
            metadata={
                "level": "segment",
                "code": row["segment"],
                "title": row["segment_title"],
            },
        )
        documents.append(doc)

    # Family level
    for _, row in df[pd.isna(df["class"]) & ~pd.isna(df["family"])].iterrows():
        doc = Document(
            text=f"FAMILY: {row['family_title']}\nDEFINITION: {row['family_definition']}\nPARENT SEGMENT: {row['segment_title']}",
            metadata={
                "level": "family",
                "code": row["family"],
                "title": row["family_title"],
                "parent_code": row["segment"],
            },
        )
        documents.append(doc)

    # Class level
    for _, row in df[pd.isna(df["commodity"]) & ~pd.isna(df["class"])].iterrows():
        doc = Document(
            text=f"CLASS: {row['class_title']}\nDEFINITION: {row['class_definition']}\nPARENT FAMILY: {row['family_title']}",
            metadata={
                "level": "class",
                "code": row["class"],
                "title": row["class_title"],
                "parent_code": row["family"],
            },
        )
        documents.append(doc)

    # Commodity level
    for _, row in df[~pd.isna(df["commodity"])].iterrows():
        doc = Document(
            text=f"COMMODITY: {row['commodity_title']}\nDEFINITION: {row['commodity_definition']}\nPARENT CLASS: {row['class_title']}",
            metadata={
                "level": "commodity",
                "code": row["commodity"],
                "title": row["commodity_title"],
                "parent_code": row["class"],
            },
        )
        documents.append(doc)

    return documents


# Create vector store index
documents = create_hierarchy_documents(data)
embed_model = OpenAIEmbedding()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever


class UNSPSCClassifier:
    def __init__(self, index: VectorStoreIndex, llm: OpenAI):
        self.index = index
        self.llm = llm
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        self.response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", llm=self.llm
        )

    def clean_description(self, description: str) -> str:
        """Clean and enrich the input description."""
        prompt = f"""Clean and standardize the following product description while preserving its meaning:
        {description}
        
        Rules:
        1. Remove unnecessary punctuation and formatting
        2. Standardize terminology
        3. Keep key product characteristics
        4. Do not add or remove important information
        """

        response = self.llm.complete(prompt)
        return response.text.strip()

    def get_confidence_score(
        self, matches, description: str, level: str
    ) -> ConfidenceScore:
        """Calculate confidence score for a particular hierarchy level."""
        prompt = f"""Given the following matches at the {level} level for the description: "{description}"
        
        Matches:
        {matches}
        
        Calculate a confidence score between 0 and 1, and provide an explanation.
        Return in format:
        Score: <score>
        Explanation: <explanation>
        """

        response = self.llm.complete(prompt)
        lines = response.text.strip().split("\n")
        score = float(lines[0].split(": ")[1])
        explanation = lines[1].split(": ")[1]

        return ConfidenceScore(score=score, explanation=explanation)

    def classify(self, description: str) -> UNSPSCMatch:
        # Clean description
        cleaned_description = self.clean_description(description)

        # Retrieve similar nodes
        nodes = self.retriever.retrieve(cleaned_description)

        # Group nodes by level
        level_nodes = {"segment": [], "family": [], "class": [], "commodity": []}

        for node in nodes:
            level = node.metadata["level"]
            level_nodes[level].append(node)

        # Get best matches and confidence scores for each level
        match = UNSPSCMatch(
            cleaned_description=cleaned_description, overall_explanation=""
        )

        # Process each level
        for level in ["segment", "family", "class", "commodity"]:
            if level_nodes[level]:
                best_node = level_nodes[level][0]
                confidence = self.get_confidence_score(
                    [n.text for n in level_nodes[level]], cleaned_description, level
                )

                setattr(match, level, best_node.metadata["code"])
                setattr(match, f"{level}_title", best_node.metadata["title"])
                setattr(match, f"{level}_confidence", confidence)

        # Generate overall explanation
        match.overall_explanation = self.response_synthesizer.synthesize(
            query=f"Explain why this product: '{cleaned_description}' was classified as shown above",
            nodes=nodes,
        ).response

        return match


# Initialize classifier
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="llama3.1:8b",
    temperature=0.7,
    max_tokens=512,
    context_window=4096,
    request_timeout=300.0,
)
classifier = UNSPSCClassifier(index, llm)

# Example classification
description = "Lincomycin antibiotic powder, 500mg, for bacterial infections"
result = classifier.classify(description)
print(result.json(indent=2))
