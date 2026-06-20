from langchain_community.document_loaders import PyPDFLoader

# pip install pypdf
loader = PyPDFLoader(
    file_path="./data/claudecode.pdf",
    mode="page",  # page模式：每页一个Document；single模式：全部页面合并为1个Document
    # password="",
)

i = 0
# 懒加载迭代读取文档
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("=" * 10, i)
