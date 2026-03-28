from prompt_template import system_template_text, user_template_text
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from xiaohongshu_model import Xiaohongshu
from dotenv import load_dotenv
import os
load_dotenv()
def generate_xiaohongshu(theme):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template_text),
        ("user", user_template_text)
    ])
    models = ChatOpenAI(
        model=os.getenv("DOUBAO_MODEL"),
        api_key=os.getenv("DOUBAO_API"),
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    output_parser = PydanticOutputParser(pydantic_object=Xiaohongshu)
    chain = prompt | models | output_parser
    result = chain.invoke({
        "parser_instructions": output_parser.get_format_instructions(),
        "theme": theme
    })
    return result
#print(generate_xiaohongshu("大模型"))

