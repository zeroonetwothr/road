import os
import pickle
import faiss
import re

from config import VECTOR_STORE_DIR
from utils.embedder import embed_texts


def extract_keywords(query: str) -> list[str]:
    """
    从用户问题中提取简单关键词，用于重排。
    """
    keywords = []

    # 提取“第七条 / 第五条 / 第三条”这类模式
    article_matches = re.findall(r'第[一二三四五六七八九十百0-9]+条', query)
    keywords.extend(article_matches)

    # 提取常见结构词
    common_terms = [
        "协议", "条款", "违约", "责任", "公务员", "升学", "入伍",
        "招聘", "简章", "荣誉", "获得", "奖项", "称号", "认证",
        "高新技术企业", "专精特新", "瞪羚", "荣誉称号", "利尔达"
    ]
    for term in common_terms:
        if term in query:
            keywords.append(term)

    return list(set(keywords))


def keyword_score(query: str, chunk: dict) -> int:
    """
    对候选 chunk 做简单关键词打分。
    """
    text = chunk["text"]
    file_name = chunk.get("file_name", "")
    keywords = extract_keywords(query)

    score = 0

    for kw in keywords:
        if kw in text:
            score += 3
        if kw in file_name:
            score += 2

    # 针对“第七条/第几条”问题，额外给含“七、”“第七条”等片段加分
    if "第七条" in query:
        if "第七条" in text or "七、" in text:
            score += 8

    if "协议" in query and "协议" in file_name:
        score += 4
    if "利尔达" in query and "利尔达" in file_name:
        score += 8

    if "荣誉" in query:
        if "荣誉" in text or "荣誉" in file_name:
            score += 6

    if "获得" in query:
        if "获得" in text:
            score += 4

    if "奖项" in query or "称号" in query:
        if "奖项" in text or "称号" in text:
            score += 4

    return score


def build_vector_store(chunks: list[dict]):
    """
    根据 chunk 元数据构建或追加向量索引，并保存 chunk 信息。
    """
    text_list = [chunk["text"] for chunk in chunks]
    new_vectors = embed_texts(text_list).astype("float32")

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    index_path = os.path.join(VECTOR_STORE_DIR, "faiss.index")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)

        index.add(new_vectors)
        all_chunks = existing_chunks + chunks
    else:
        dimension = new_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(new_vectors)
        all_chunks = chunks

    faiss.write_index(index, index_path)

    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)


def search_similar_chunks(query: str, top_k: int = 3, recall_k: int = 20) -> list[dict]:
    """
    先用向量检索召回，再用关键词做简单重排。
    """
    index_path = os.path.join(VECTOR_STORE_DIR, "faiss.index")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return []

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    query_vector = embed_texts([query]).astype("float32")
    distances, indices = index.search(query_vector, recall_k)

    candidates = []
    for rank, idx in enumerate(indices[0]):
        if idx != -1:
            chunk = chunks[idx]
            score = keyword_score(query, chunk)

            candidates.append({
                "chunk": chunk,
                "vector_rank": rank,
                "keyword_score": score
            })

    # 重排规则：先按关键词分数降序，再按原始向量排名升序
    candidates.sort(key=lambda x: (-x["keyword_score"], x["vector_rank"]))

    results = [item["chunk"] for item in candidates[:top_k]]
    return results