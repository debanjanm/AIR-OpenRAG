from typing import List
import asyncio
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

class OpenRouterEmbeddings(Embeddings):
    """Custom embedding class for OpenRouter API with async support"""
    
    def __init__(self, api_key: str, model: str = "qwen/qwen3-embedding-8b", 
                 site_url: str = None, site_name: str = None, 
                 show_progress: bool = True, max_concurrent: int = 10):
        # Sync client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        # Async client
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.show_progress = show_progress
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (sync version)"""
        embeddings = []
        iterator = tqdm(texts, desc="Embedding documents") if self.show_progress else texts
        
        for text in iterator:
            response = self.client.embeddings.create(
                extra_headers=self.extra_headers,
                model=self.model,
                input=text,
                encoding_format="float"
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    async def _embed_single(self, text: str) -> List[float]:
        """Embed a single text asynchronously with rate limiting"""
        async with self.semaphore:
            response = await self.async_client.embeddings.create(
                extra_headers=self.extra_headers,
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents asynchronously with concurrent requests"""
        tasks = [self._embed_single(text) for text in texts]
        
        if self.show_progress:
            embeddings = await async_tqdm.gather(*tasks, desc="Embedding documents")
        else:
            embeddings = await asyncio.gather(*tasks)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (sync)"""
        response = self.client.embeddings.create(
            extra_headers=self.extra_headers,
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query asynchronously"""
        return await self._embed_single(text)
