from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# Chroma 轻量级向量数据库
# pip install langchain-chroma chromadb dashscope

# 初始化向量库
vector_store = Chroma(
    collection_name="test",        # 向量集合名称，类似数据表名
    embedding_function=DashScopeEmbeddings(),  # 文本嵌入模型
    persist_directory="./chroma_db"# 向量数据持久化存储目录
)

# 加载CSV文档
loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
    source_column="source"  # 指定作为文档来源元数据的列
)
documents = loader.load()

# 向向量库新增文档，自定义每条文档id
vector_store.add_documents(
    documents=documents,
    ids=["id" + str(i) for i in range(1, len(documents) + 1)]
)

# 根据id删除指定向量文档
vector_store.delete(["id1", "id2"])

# 语义搜索 - 提问内容与数据不完全相等，但具有相关性
result = vector_store.similarity_search("使用Python处理数据的专家", k=3)
print("搜索结果：", result)

result2 = vector_store.similarity_search("推荐系统算法", k=2)
print("搜索结果2：", result2)

result3 = vector_store.similarity_search("自动化测试框架", k=2)
print("搜索结果3：", result3)
