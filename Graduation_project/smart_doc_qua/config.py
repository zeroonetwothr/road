import os
from dotenv import load_dotenv
load_dotenv(".environment")
DOCS_DIR = "data/docs"
VECTOR_STORE_DIR = "vector_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
LLM_API_KEY = os.getenv("ARK_API_KEY")
LLM_MODEL_NAME = "doubao-seed-2-0-mini-260215"
PROMPT_VERSION = "v3"