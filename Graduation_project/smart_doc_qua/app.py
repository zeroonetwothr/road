import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DOCS_DIR
from utils.pdf_loader import load_pdf_pages
from utils.text_splitter import split_pages_into_chunks
from utils.retriever import build_vector_store, search_similar_chunks
from utils.llm import ask_llm

app = FastAPI(title="基于提示工程和微调的智能文档问答助手")

os.makedirs("static", exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(files: list[UploadFile] = File(...)):
    """
    上传多个 PDF 文档，按页提取文本、切分文本并建立向量索引。
    """
    all_chunks = []

    for file in files:
        file_path = os.path.join(DOCS_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        pages = load_pdf_pages(file_path)
        chunks = split_pages_into_chunks(pages, file.filename)
        all_chunks.extend(chunks)

    build_vector_store(all_chunks)

    return {
        "message": "多个文档上传并建库成功",
        "file_count": len(files),
        "chunk_count": len(all_chunks)
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    根据用户问题和历史对话进行向量检索，并调用大模型生成回答。
    """
    question = request.question
    history = request.history

    context_chunks = search_similar_chunks(question, top_k=3)
    llm_result = ask_llm(question, context_chunks, history)

    return {
        "question": question,
        "history": history,
        "retrieved_chunks": context_chunks,
        "answer": llm_result["answer"],
        "llm_success": llm_result["success"],
        "error": llm_result["error"]
    }