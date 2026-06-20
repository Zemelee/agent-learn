from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="./data/stu.json",
    jq_schema=".[]", # .[] 表示所有对象
    text_content=False # 抽取的内容不是字符串
)
documents = loader.load()
print(documents)
print("\n")

loader = JSONLoader(
    file_path="./data/stu.json",
    jq_schema=".[].score", # 抽取的json对象
    text_content=False # 抽取的内容不是字符串
)
documents = loader.load()
print(documents)