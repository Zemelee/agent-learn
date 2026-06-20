from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough


# 初始化通义大模型
model = ChatTongyi(model="qwen3-max")

# 构建对话提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "{input}")
    ]
)

# 初始化内存向量库，使用阿里通义嵌入模型
vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 向向量库添加文本数据
vector_store.add_texts([
    "减肥就是要少吃多练",
    "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来",
    "跑步是很好的运动哦"
])

# 用户输入问题
input_text = "怎么减肥？"

# 相似度检索，取出Top2相关文档
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def format_func(docs: List[Document]):
    if not docs:
        return "无相关参考资料"
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"
    return formatted_str

# 构建RAG完整执行链
chain = (
    {
        "input": RunnablePassthrough(), # 输入截流
        "context": retriever | format_func
    }
    | prompt
    | model
    | StrOutputParser()
)

# 执行调用
res = chain.invoke(input_text)
print(res)

"""
流程说明：
retriever:
  - 输入：用户的提问 str
  - 输出：向量库的检索结果 list[Document]
prompt:
  - 输入：用户的提问 + 向量库的检索结果 dict
  - 输出：完整的提示词 PromptValue
"""
