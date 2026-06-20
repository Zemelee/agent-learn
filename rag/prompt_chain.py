# import langchain
# print(langchain.__version__) # 1.2.13

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

chat = ChatTongyi(model="qwen3.6-max-preview")
# res = chat.invoke("你是谁？") # AIMessage
# print(res.content) # 我是 Qwen...
# print(res.type) # ai
# print(res.additional_kwargs) # {model_name,finish_reason,request_id,,token_usage:{}}
# print(res.response_metadata)

msg = [
    SystemMessage(content="你是个没用的ai"),
    HumanMessage(content="你好"),
    AIMessage(content="我是个没用的ai."),
    HumanMessage(content="为什么这么说？"),
]
# msg = [
#     ("system", "你是个没用的ai."),
#     ("human", "你好"),
#     ("ai", "我是个没用的ai."),
#     ("human", "为什么这么说？"),
# ]
# res = chat.stream(input=msg)
# for chunk in res:
#     print(chunk.content, end="", flush=True)

# 导入  少量样本提示词模板、基础提示词模板
from langchain_core.prompts import  PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
"""
PromptTemplate-->StringPromptTemplate-->BasePromptTemplate-->Runnable
FewShotPromptTemplate-->StringPromptTemplate-->BasePromptTemplate-->Runnable
ChatPromptTemplate-->BasePromptTemplate-->Runnable

PromptTemplate：通用提示词模板，支持动态注入信息
FewShotPromptTemplate： 可注入任意数量的示例信息
ChatPromptTemplate：可注入任意数量的历史会话信息
"""



# ============================
# PromptTemplate
# ============================
# 多变量提示词模板
template1 = PromptTemplate(
    template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。",
    input_variables=["product", "aspect1", "aspect2"],
)
prompt_1 = template1.format(product="智能手机", aspect1="电池续航", aspect2="拍照质量")
print("提示词1:", prompt_1)



# 定义多变量模板
template = PromptTemplate(
    template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。",
    input_variables=["product", "aspect1", "aspect2"],
    # partial_variables={"aspect1":"电池续航","aspect2":"拍照质量"}
)
# partial生成新模板
template3 = template.partial(aspect1="电池续航", aspect2="拍照质量")
prompt_3 = template3.format(product="智能手机")
print("提示词3:", prompt_3)

# 组合提示词
template4 = PromptTemplate.from_template(
    "Tell me a joke about {topic}" + ", make it funny" + "and in {language}"
)
prompt_4 = template4.format(topic="sports", language="spanish")
print("提示词4:", prompt_4)


prompt_1 = template.invoke(
    input={"product": "智能手机", "aspect1": "电池续航", "aspect2": "拍照质量"}
)
print(prompt_1)
print(type(prompt_1))  # StringPromptValue
print("提示词5:", prompt_1.to_string())


template = PromptTemplate.from_template("我的邻居是: {lastname}，最喜欢: {hobby}")
res = template.format(lastname="张大明", hobby="钓鱼")
print(res, type(res)) # <class 'str'>

res2 = template.invoke({"lastname": "周杰伦", "hobby": "唱歌"})
print(res2, type(res2)) # <class 'langchain_core.prompts.prompt.PromptValue'>

# ============================
# ChatPromptTemplate
# ============================


# 创建对话提示词模板
chat_prompt_template = ChatPromptTemplate(
    messages=[
        ("system", "你是一个AI助手，你的名字叫{name}"),
        ("human", "我的问题是{question}"),
    ],
    # input_variables=["name", "question"], # 可删除
)

chat_prompt_template = ChatPromptTemplate(
    [("system", "你是一个AI助手，你的名字叫{name}"), ("human", "我的问题是{question}")]
)

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你的名字叫{name}"),
    ("human", "我的问题是{question}")
])

# 填充变量生成对话消息
response = chat_prompt_template.invoke(
    input={"name": "小智", "question": "1 + 2 * 3 = ?"}
)

print(response)
print(type(response))  # <class 'langchain_core.prompt_values.ChatPromptValue'>
print(len(response.messages))  # 2

# ============================
# FewShotPromptTemplate
# ============================

example_template = PromptTemplate.from_template("单词:{word}，反义词:{antonym}")

example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data, # 示例数据,list
    prefix="给出给定词的反义词，有如下示例：",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    # 声明前后缀里用到的动态变量名
    input_variables=['input_word']
)

prompt_text = few_shot_prompt.invoke(input={"input_word": "左"})
print(prompt_text.to_string())
print("\n\n")
# 给出给定词的反义词，有如下示例：
# 单词:大，反义词:小
# 单词:上，反义词:下
# 基于示例告诉我：左的反义词是？
# print(chat.invoke(input=prompt_text))


from langchain_core.prompts import MessagesPlaceholder

# 构建对话模板：系统角色 + 历史对话占位符 + 当前用户提问
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),  # 历史对话占位，变量名history
        ("human", "请再来一首唐诗"),
    ]
)

# 历史对话记录：多轮 human + ai 成对消息
history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]
# 填充history变量，渲染完整对话，转为纯文本字符串
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
# print(prompt_text)

# | 的元素必须是 Runnable 的子类
# chain = chat_prompt_template | chat # <RunnableSequence>
# res = chain.stream(input=msg)
# for chunk in res:
#     print(chunk.content, end="", flush=True)


# res = chat.invoke(prompt_text) # <AIMessage>
# print(res.content)


from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
prompt1 = PromptTemplate.from_template(
    "我邻居姓：{last_name}，刚生了{gender}，请起名，仅需告诉我名字，无需告诉我其他内容。"
)
# 不标准的链条
chain = prompt1 | chat | parser | chat | parser
# res = chain.invoke({"last_name": "张", "gender": "女儿"})
# print(res)




from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# 1. 定义两种输出解析器
str_parser = StrOutputParser()    # 字符串解析器：直接返回文本字符串
json_parser = JsonOutputParser()  # JSON解析器：将模型返回文本转为字典
new_dict = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，仅告诉我姓名，不要输出其他"
)
second_prompt = PromptTemplate.from_template("{name}，你觉得这个名字怎么样？")
chain = first_prompt | chat | new_dict | second_prompt | chat | str_parser

# 6. 传入初始参数执行整条链路
res: str = chain.stream({"lastname": "张", "gender": "女儿"})
for chunk in res:
    print(chunk)

