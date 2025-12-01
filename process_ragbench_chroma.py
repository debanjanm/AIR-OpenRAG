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
# Create Chroma vector store with persistence
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     persist_directory="./chroma_ragbench_db",
#     collection_name="open_ragbench"
# )

# print(f"Stored {vectorstore._collection.count()} document chunks")

##------------------------------------------------------------------------------------##
# Load existing vector store
vectorstore = Chroma(
    persist_directory="./chroma_ragbench_db",
    embedding_function=embeddings,
    collection_name="open_ragbench"
)
##------------------------------------------------------------------------------------##
# Basic similarity search
query = "What are the challenges in estimating output impedance in inverter-based grids?"
# results = vectorstore.similarity_search(
#     query=query,
#     k=20  # Return top 5 results
# )

# # # Access results with metadata
# for doc in results:
#     print(f"Title: {doc.metadata['title']}")
#     print(f"Metadata: {doc.metadata}")
#     # print(f"Section: {doc.metadata['section']}")
#     # print(f"Content: {doc.page_content[:500]}...")
#     # print(f"Authors: {doc.metadata['authors']}")
#     print("---")
##------------------------------------------------------------------------------------##
# Similarity search with scores
# results_with_scores = vectorstore.similarity_search_with_relevance_scores(
#     query=query,
#     k=20
# )

# for doc, score in results_with_scores:
#     print(f"Score: {score}")
#     print(f"Title: {doc.metadata['title']}")
#     print(f"Content preview: {doc.page_content[:100]}...")
##------------------------------------------------------------------------------------##
# # Filtered search by metadata
results_filtered = vectorstore.similarity_search(
    query=query,
    k=5,
    filter={"paper_id": "2410.14077v2"}  
)

print(f"Found {len(results_filtered)} results with tables.")

for doc in results_filtered:
    print(f"Title: {doc.metadata['title']}")
    print(f"Section: {doc.metadata['section']}")
    print(f"Has Tables: {doc.metadata['has_tables']}")
    print(f"Content preview: {doc.page_content[:100]}...")
    print("===")
##------------------------------------------------------------------------------------##

