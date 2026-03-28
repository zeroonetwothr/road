from openai import OpenAI
from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME, PROMPT_VERSION


def format_history(history: list[dict]) -> str:
    """
    将历史对话整理成字符串，方便拼接到提示词中。
    """
    if not history:
        return "无历史对话"

    history_lines = []

    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")

        if role == "user":
            history_lines.append(f"用户：{content}")
        elif role == "assistant":
            history_lines.append(f"助手：{content}")

    return "\n".join(history_lines)


def format_context(context_chunks: list[dict]) -> str:
    """
    将检索到的文本块整理成统一格式的上下文字符串。
    """
    context_parts = []

    for chunk in context_chunks[:3]:
        file_name = chunk.get("file_name", "未知文档")
        page_num = chunk.get("page", "未知页码")
        chunk_text = chunk.get("text", "")

        context_parts.append(
            f"【来源文档: {file_name} | 来源页码: 第{page_num}页】\n{chunk_text}"
        )

    return "\n\n".join(context_parts)


def build_prompt_v1(context_chunks: list[dict], question: str) -> str:
    """
    Prompt V1：基础约束版
    目标：严格基于文档回答，不要编造。
    """
    context = format_context(context_chunks)

    prompt = f"""
你是一个智能文档问答助手。
请严格依据提供的文档内容回答问题，不要编造信息。

要求：
1. 如果答案可以从文档中找到，请直接回答。
2. 如果文档中找不到答案，请明确回答：根据当前提供的文档内容，无法确定该问题的答案。
3. 不要使用文档之外的常识来补充答案。

文档内容：
{context}

用户问题：
{question}

请开始回答：
"""
    return prompt


def build_prompt_v2(context_chunks: list[dict], question: str) -> str:
    """
    Prompt V2：来源增强版
    目标：在回答基础上尽量标注文档来源和页码。
    """
    context = format_context(context_chunks)

    prompt = f"""
你是一个智能文档问答助手。
请严格依据提供的文档内容回答问题，不要编造信息。

要求：
1. 如果答案可以从文档中找到，请用简洁、清晰的语言回答。
2. 如果文档中找不到答案，请明确回答：根据当前提供的文档内容，无法确定该问题的答案。
3. 回答时尽量引用文档中的关键信息。
4. 如果能够判断来源，请尽量标注文档名称和页码。
5. 不要使用文档之外的常识来补充答案。

文档内容：
{context}

用户问题：
{question}

请开始回答：
"""
    return prompt


def build_prompt_v3(context_chunks: list[dict], question: str, history: list[dict]) -> str:
    """
    Prompt V3：多轮理解增强版
    目标：结合历史对话理解上下文，并标注来源。
    """
    context = format_context(context_chunks)
    history_text = format_history(history)

    prompt = f"""
你是一个智能文档问答助手。
请严格依据提供的文档内容回答问题，不要编造信息。

要求：
1. 如果答案可以从文档中找到，请用简洁、清晰的语言回答。
2. 如果文档中找不到答案，请明确回答：根据当前提供的文档内容，无法确定该问题的答案。
3. 回答时尽量引用文档中的关键信息。
4. 如果能够判断来源，请尽量标注文档名称和页码。
5. 可以参考历史对话来理解用户当前问题中的代词、省略和上下文。
6. 不要使用文档之外的常识来补充答案。

历史对话：
{history_text}

文档内容：
{context}

用户当前问题：
{question}

请开始回答：
"""
    return prompt


def build_prompt(context_chunks: list[dict], question: str, history: list[dict]) -> str:
    """
    根据配置选择不同版本的提示模板。
    """
    if PROMPT_VERSION == "v1":
        return build_prompt_v1(context_chunks, question)
    elif PROMPT_VERSION == "v2":
        return build_prompt_v2(context_chunks, question)
    else:
        return build_prompt_v3(context_chunks, question, history)


def build_fallback_answer(question: str, context_chunks: list[dict]) -> str:
    """
    当大模型调用失败时，返回一个基于检索结果的兜底回答。
    """
    if not context_chunks:
        return "未找到相关文档内容，暂时无法回答。"

    lines = []

    for chunk in context_chunks[:3]:
        file_name = chunk["file_name"]
        page_num = chunk["page"]
        chunk_text = chunk["text"]

        lines.append(f"【{file_name} 第{page_num}页】{chunk_text}")

    joined = "\n\n".join(lines)

    return f"""模型暂时不可用，以下是检索到的相关文档内容，可供参考：

问题：{question}

相关内容：
{joined[:1200]}
"""


def ask_llm(question: str, context_chunks: list[dict], history: list[dict]) -> dict:
    """
    使用大模型进行问答，返回统一结构。
    """
    if not context_chunks:
        return {
            "success": False,
            "answer": "未找到相关文档内容，暂时无法回答。",
            "error": "no_context"
        }

    prompt = build_prompt(context_chunks, question, history)

    try:
        client = OpenAI(
            base_url=LLM_API_URL,
            api_key=LLM_API_KEY,
        )

        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的文档问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        return {
            "success": True,
            "answer": completion.choices[0].message.content,
            "error": None
        }

    except Exception as e:
        fallback_answer = build_fallback_answer(question, context_chunks)

        return {
            "success": False,
            "answer": fallback_answer,
            "error": str(e)
        }