from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]):
    """
    将多个文本块转换为向量表示。
    """
    return model.encode(texts, convert_to_numpy=True)