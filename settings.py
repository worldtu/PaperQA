"""
Configuration settings for the PaperQA system.
Contains model names, batch sizes, MMR parameters, and thresholds.
"""

import os
from typing import Dict, Any
from schemas import Config


class Settings:
    """Centralized settings for the PaperQA system."""
    
    # Model Configuration
    MODEL_NAME: str = "Qwen/Qwen3-0.6B"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # System Prompts
    SYSTEM_PROMPT_LLM: str = """
Answer in an direct and concise tone, I am in a hurry. Your audience is an expert, so be highly specific. If there are ambiguous terms or acronyms, first define them.
"""
    
    SYSTEM_PROMPT_AGENT: str = """
You are a helpful AI assistant.
"""
    
    # Data Paths
    PAPERS_FOLDER: str = "./papers"
    DATA_DIR: str = "./data"
    CACHE_DIR: str = "./data/cache"
    VECTOR_INDEX_PATH: str = "./data/faiss.index"
    SQLITE_DB_PATH: str = "./data/store.sqlite"
    
    # RAG Configuration
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 800
    
    # Search Configuration
    DEFAULT_K: int = 5
    DEFAULT_FETCH_K: int = 20
    DEFAULT_LAMBDA_MULT: float = 0.5
    
    # LLM Configuration
    DEFAULT_TEMPERATURE: float = 0.6
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_TOP_K: int = 20
    DEFAULT_MIN_P: float = 0.0
    DEFAULT_PRESENCE_PENALTY: float = 1.5
    MAX_NEW_TOKENS: int = 32768
    
    # Answer Configuration
    DEFAULT_ANSWER_LENGTH: int = 512
    DEFAULT_SUMMARY_LENGTH: int = 500
    DEFAULT_TOP_CHUNK_NUM: int = 3
    
    # Evaluation Configuration
    EVALUATION_DATASET: str = "futurehouse/lab-bench"
    EVALUATION_SPLIT: str = "LitQA2"
    
    # Agent Configuration
    MIN_EVIDENCE_COUNT: int = 5
    MAX_ITERATIONS: int = 10
    
    @classmethod
    def get_config(cls) -> Config:
        """Get a Config object with current settings."""
        return Config(
            answer_length=cls.DEFAULT_ANSWER_LENGTH,
            summary_length=cls.DEFAULT_SUMMARY_LENGTH,
            top_chunk_num=cls.DEFAULT_TOP_CHUNK_NUM,
            k=cls.DEFAULT_K,
            fetch_k=cls.DEFAULT_FETCH_K,
            lambda_mult=cls.DEFAULT_LAMBDA_MULT,
            temperature=cls.DEFAULT_TEMPERATURE,
            top_p=cls.DEFAULT_TOP_P,
            top_k=cls.DEFAULT_TOP_K,
            min_p=cls.DEFAULT_MIN_P,
            presence_penalty=cls.DEFAULT_PRESENCE_PENALTY
        )
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.PAPERS_FOLDER, exist_ok=True)
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            "model_name": cls.MODEL_NAME,
            "embedding_model": cls.EMBEDDING_MODEL,
            "dtype": "auto",
            "device_map": "auto"
        }
    
    @classmethod
    def get_rag_config(cls) -> Dict[str, Any]:
        """Get RAG configuration parameters."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "pdf_folder": cls.PAPERS_FOLDER
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration parameters."""
        return {
            "k": cls.DEFAULT_K,
            "fetch_k": cls.DEFAULT_FETCH_K,
            "lambda_mult": cls.DEFAULT_LAMBDA_MULT
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM generation configuration parameters."""
        return {
            "temperature": cls.DEFAULT_TEMPERATURE,
            "top_p": cls.DEFAULT_TOP_P,
            "top_k": cls.DEFAULT_TOP_K,
            "min_p": cls.DEFAULT_MIN_P,
            "presence_penalty": cls.DEFAULT_PRESENCE_PENALTY,
            "max_new_tokens": cls.MAX_NEW_TOKENS,
            "enable_thinking": True
        }
