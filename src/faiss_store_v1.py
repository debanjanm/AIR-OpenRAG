import os
import faiss
import pickle
import numpy as np
from typing import List, Optional, Tuple, Union

# Replace this import with whatever Document class you use.
# I keep it generic: a document is an object with `.page_content` and optional `.metadata`.
# If you use langchain_core.documents.Document, import that class and pass Documents.
try:
    from langchain_core.documents import Document  # type: ignore
except Exception:
    # Fallback simple Document dataclass if langchain_core not available
    from dataclasses import dataclass
    @dataclass
    class Document:
        page_content: str
        metadata: dict = None

class FaissCosineStore:
    """FAISS-based vector store using cosine similarity (via normalized vectors + IP index).
    
    Usage:
        store = FaissCosineStore(embedding_dim)
        store.build_from_documents(docs, embeddings)
        store.save(path)
        # later...
        store = FaissCosineStore.load(path, embeddings=embeddings)
        results = store.similarity_search("my query", k=5)
    """

    INDEX_FILENAME = "faiss.index"
    METADATA_FILENAME = "documents.pkl"
    CONFIG_FILENAME = "store_config.pkl"

    def __init__(self, embedding_dim: int, index: Optional[faiss.Index] = None):
        self.embedding_dim = embedding_dim
        # Use inner-product index; we'll L2-normalize vectors so inner product == cosine
        if index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = index
        self._documents: List[Document] = []
        # total number of vectors in index
        self.ntotal = int(self.index.ntotal) if hasattr(self.index, "ntotal") else 0

    @staticmethod
    def _ensure_doc(doc_or_text: Union[Document, str]) -> Document:
        if isinstance(doc_or_text, Document):
            return doc_or_text
        else:
            return Document(page_content=str(doc_or_text), metadata={})

    @staticmethod
    def _normalize_np_array(arr: np.ndarray) -> np.ndarray:
        """L2-normalize rows. Expects float32 array shaped (n, d)."""
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # avoid divide by zero
        norms[norms == 0.0] = 1.0
        arr = arr / norms
        return arr

    def build_from_documents(self, docs: List[Union[Document, str]], embeddings) -> None:
        """Create index from list of Document or strings using `embeddings.embed_documents`."""
        docs = [self._ensure_doc(d) for d in docs]
        texts = [d.page_content for d in docs]

        # get vectors from embeddings object
        vecs = embeddings.embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)

        if vecs.ndim != 2 or vecs.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings shape mismatch. Expected (N, {self.embedding_dim}), got {vecs.shape}.")

        vecs = self._normalize_np_array(vecs)
        # add to index
        self.index.add(vecs)
        # store docs in same order for mapping indices -> docs
        self._documents = docs
        self.ntotal = int(self.index.ntotal)

    def add_documents(self, docs: List[Union[Document, str]], embeddings) -> None:
        """Add more documents (incremental)."""
        docs = [self._ensure_doc(d) for d in docs]
        texts = [d.page_content for d in docs]
        vecs = embeddings.embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings shape mismatch. Expected (N, {self.embedding_dim}), got {vecs.shape}.")
        vecs = self._normalize_np_array(vecs)
        self.index.add(vecs)
        self._documents.extend(docs)
        self.ntotal = int(self.index.ntotal)

    def similarity_search(self, query: str, embeddings, k: int = 4) -> List[Tuple[Document, float]]:
        """Return top-k (Document, cosine_score) for the given query string."""
        q_vec = embeddings.embed_query(query)
        q_vec = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
        if q_vec.shape[1] != self.embedding_dim:
            raise ValueError(f"Query embedding dim mismatch: expected {self.embedding_dim}, got {q_vec.shape[1]}")

        q_vec = self._normalize_np_array(q_vec)  # normalize
        # search using inner product => returns scores = inner products == cosine similarity
        distances, indices = self.index.search(q_vec, k)
        distances = distances.flatten()
        indices = indices.flatten()

        results: List[Tuple[Document, float]] = []
        for idx, score in zip(indices, distances):
            if idx < 0 or idx >= len(self._documents):
                continue
            results.append((self._documents[int(idx)], float(score)))
        return results

    def similarity_search_with_relevance_scores(self, query: str, embeddings, k: int = 4) -> List[Tuple[Document, float]]:
        """Alias for compatibility: returns (doc, score) with score as cosine similarity."""
        return self.similarity_search(query, embeddings, k)

    def save(self, persist_directory: str) -> None:
        """Save FAISS index and documents to disk."""
        os.makedirs(persist_directory, exist_ok=True)
        index_path = os.path.join(persist_directory, self.INDEX_FILENAME)
        meta_path = os.path.join(persist_directory, self.METADATA_FILENAME)
        config_path = os.path.join(persist_directory, self.CONFIG_FILENAME)

        # write faiss index binary
        faiss.write_index(self.index, index_path)

        # save documents metadata and page_content in pickle (keeps order)
        with open(meta_path, "wb") as f:
            pickle.dump(self._documents, f)

        # save some config (embedding_dim)
        with open(config_path, "wb") as f:
            pickle.dump({"embedding_dim": self.embedding_dim}, f)

    @classmethod
    def load(cls, persist_directory: str, embeddings=None) -> "FaissCosineStore":
        """Load store from disk. `embeddings` is optional (only required for future queries/adding)."""
        index_path = os.path.join(persist_directory, cls.INDEX_FILENAME)
        meta_path = os.path.join(persist_directory, cls.METADATA_FILENAME)
        config_path = os.path.join(persist_directory, cls.CONFIG_FILENAME)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            docs = pickle.load(f)
        with open(config_path, "rb") as f:
            cfg = pickle.load(f)

        embedding_dim = int(cfg["embedding_dim"])
        store = cls(embedding_dim=embedding_dim, index=index)
        store._documents = docs
        store.ntotal = int(store.index.ntotal)
        # embeddings object can be passed optionally for query/adding later
        store._embeddings = embeddings
        return store

    # convenience wrappers if you passed embeddings to load()
    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        if getattr(self, "_embeddings", None) is None:
            raise RuntimeError("No embeddings object attached. Pass embeddings to load() or use similarity_search with embeddings.")
        return self.similarity_search(query, self._embeddings, k)
