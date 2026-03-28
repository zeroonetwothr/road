import PyPDF2


def load_pdf_pages(file_path: str) -> list[dict]:
    """
    读取 PDF 文件，按页返回文本内容。
    每一页保存为一个字典，包含页码和文本。
    """
    pages = []

    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for page_index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""

            pages.append({
                "page": page_index + 1,
                "text": page_text
            })

    return pages