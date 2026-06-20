# LangChain 学习项目

一个用于学习 LangChain 的初步尝试。

## 环境依赖

- Python 3.12
- langchain == 1.2.13

## 安装

```bash
export DASHSCOPE_API_KEY='sk-xxxxx' # TODO
pip install langchain==1.2.13 langchain-chroma chromadb dashscope
```

## 项目说明

该项目主要用于学习和实践 LangChain 的基本功能，包括：
- 文档加载（CSV、JSON、TXT）
- 向量数据库（Chroma）
- 嵌入模型（DashScope）
- 相似性搜索

## 文件结构

- `loader_csv.py` - CSV 文档加载示例
- `loader_json.py` - JSON 文档加载示例
- `vector.py` - 向量数据库与相似性搜索示例
- `data/` - 测试数据文件夹
