## UNSPSC Finder

Our main task here based on product categories try to classify them using UNSPSC list which used for generally in such purposes. For this, we implement RAG+LLM approach which use all our UNSPSC list and hieriracally find most liked on using SentenceTransformer and using HNSWLIB vector db for store them in fast retrieving. As model, we used GROQ FREE API which tested with differrent models which help us identify that for best one is Mistral.


## Our Runtime
![Workflow Example](/images/runtime-image.png)
