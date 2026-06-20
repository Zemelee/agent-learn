from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="./data/stu.csv",
    csv_args={
        "delimiter": ",",    # 指定分隔符
        "quotechar": "'",   # 指定带有分隔符文本的引号包围是单引号
    },
    encoding="utf-8"        # 指定编码为UTF-8
)

# 批量加载 .load() -> [Document, Document, ...]
# documents = loader.load()
# for document in documents:
#     print(type(document), document)

# 懒加载 .lazy_load() 迭代器[Document] 适合大文档加载
for document in loader.lazy_load():
    print(document)