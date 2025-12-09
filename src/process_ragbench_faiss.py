##------------------------------------------------------------------------------------##
import json
import os
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
##------------------------------------------------------------------------------------##
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_BASE"] = os.getenv("LMSTUDIO_API_BASE")
# os.environ["OPENAI_API_KEY"] = os.getenv("LMSTUDIO_API_KEY")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")  
os.environ["OPENAI_API_BASE"] = os.getenv("OPENROUTER_API_BASE")
##------------------------------------------------------------------------------------##
model_name = "qwen/qwen3-embedding-8b"

from optimized_embed_v6 import OpenRouterEmbeddings
embeddings = OpenRouterEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    max_concurrent=1  
)
##------------------------------------------------------------------------------------##
# Load corpus from local directory
DATA_DIR = "open_ragbench_local"
corpus_path = os.path.join(DATA_DIR, "corpus")
##------------------------------------------------------------------------------------##
def load_ragbench_documents(corpus_path: str) -> List[Document]:
    """Load documents from the Open RAG Benchmark corpus."""
    documents = []
    
    for json_file in Path(corpus_path).glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        # Extract metadata
        metadata_base = {
            "title": paper_data.get("title", ""),
            "authors": ", ".join(paper_data.get("authors", [])),
            "categories": ", ".join(paper_data.get("categories", [])),
            "abstract": paper_data.get("abstract", ""),
            "published": paper_data.get("published", ""),
            "paper_id": paper_data.get("id", "")
        }
        
        # Process each section
        for section_idx, section in enumerate(paper_data.get("sections", [])):
            # Create structured chunk
            chunk_text = section.get("text", "")
            
            # Create Document with metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    **metadata_base,
                    "section": f"section_{section_idx}",
                    "section_index": section_idx,
                    "has_tables": len(section.get("tables", {})) > 0,
                    "has_images": len(section.get("images", {})) > 0
                }
            )
            documents.append(doc)
    
    return documents

# Load documents
documents = load_ragbench_documents(corpus_path)
##------------------------------------------------------------------------------------##
# len_documents = len(documents)
# print(f"Total documents to embed: {len_documents}")

# filtered_documents = []

# paper_ids_to_include = ["2410.14077v2", "2404.00822v2", "2410.07168v2", "2401.07294v4", "2411.14884v3", "2410.11774v2", "2412.02582v2", "2406.17972v3", "2410.09697v2", "2412.10128v2"]

# for i, doc in enumerate(documents):
#     if doc.metadata.get("paper_id") in paper_ids_to_include:
#         filtered_documents.append(doc)

# print(f"Total documents to embed: {len(filtered_documents)}")

##------------------------------------------------------------------------------------##
from faiss_store_v1 import FaissCosineStore

# docs = filtered_documents

# embedding_dim = 4096  # set to your embedding vector size
# store = FaissCosineStore(embedding_dim=embedding_dim)

# Build index
# store.build_from_documents(docs, embeddings)

# Save to disk
# store.save("./faiss_ragbench_store")
##------------------------------------------------------------------------------------##
# Load later (optionally attach the same embeddings object)
loaded = FaissCosineStore.load("./faiss_ragbench_store", embeddings=embeddings)
##------------------------------------------------------------------------------------##
# Search
results = loaded.similarity_search("What are the challenges in estimating output impedance in inverter-based grids?", embeddings, k=5)
for doc, score in results:
    print(score, doc.metadata["title"])
##------------------------------------------------------------------------------------##

