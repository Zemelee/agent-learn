from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 安装依赖命令
# pip install langchain_text_splitters

# 加载本地txt文档
loader = TextLoader(file_path="./data/story.txt", encoding="utf-8")
docs = loader.load()  # 返回Document对象列表

# 初始化递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 单块文本最大字符长度
    chunk_overlap=50,      # 块与块之间重叠字符数
    # 文本分割优先级分隔符，从长段落到短句依次切割
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    length_function=len    # 用于计算文本长度的函数
)

# 对加载后的文档执行分割
split_docs = text_splitter.split_documents(docs)

# 遍历打印分割后的文本块
for idx, doc in enumerate(split_docs):
    print(f"===== 第{idx+1}个文本块 =====")
    print(doc.page_content)
    print("元数据：", doc.metadata)
    print()