import re


def split_pages_into_chunks(
    pages: list[dict],
    file_name: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:
    """
    将按页读取的文本优先按段落切分，再组合成 chunk。
    这样更适合协议、简章、合同等结构化文档。
    """
    chunks = []

    for page_item in pages:
        page_num = page_item["page"]
        text = page_item["text"].strip()

        if not text:
            continue

        # 先按换行拆成段落，过滤空段
        paragraphs = [p.strip() for p in re.split(r'[\r\n]+', text) if p.strip()]

        current_chunk = ""

        for para in paragraphs:
            # 如果当前段落本身特别长，就按字符兜底切分
            if len(para) > chunk_size:
                if current_chunk:
                    chunks.append({
                        "file_name": file_name,
                        "page": page_num,
                        "text": current_chunk.strip()
                    })
                    current_chunk = ""

                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunk_text = para[start:end]

                    chunks.append({
                        "file_name": file_name,
                        "page": page_num,
                        "text": chunk_text.strip()
                    })

                    start += chunk_size - overlap

                continue

            # 如果拼上当前段落后还没超长，就继续拼
            if len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk += para + "\n"
            else:
                # 保存当前 chunk
                chunks.append({
                    "file_name": file_name,
                    "page": page_num,
                    "text": current_chunk.strip()
                })

                # 新 chunk 从当前段落开始
                current_chunk = para + "\n"

        # 别忘了收尾
        if current_chunk.strip():
            chunks.append({
                "file_name": file_name,
                "page": page_num,
                "text": current_chunk.strip()
            })

    return chunks