from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import  PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

chat = ChatTongyi(model="qwen3.6-max-preview")
def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

prompt = PromptTemplate.from_template(
    "你需要根据对话历史回应用户问题。对话历史:{chat_history}。用户当前输入: {input}，请回答"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你需要根据对话历史回应用户问题。"),
    MessagesPlaceholder("chat_history"),
    ("human", "请回答如下问题：{input}")
])


# 3. 基础执行链：模板 -> 打印调试prompt -> 模型调用 -> 字符串解析
base_chain = prompt | print_prompt | chat | StrOutputParser()

# InMemoryChatMessageHistory(messages=[HumanMessage(content='小明有一只猫'), AIMessage(content='您提供的...')])
# 4. 全局会话存储字典：key=session_id, value=InMemoryChatMessageHistory实例
chat_history_store = {}

# 根据session_id获取对应会话历史，不存在则新建内存会话
def get_history(session_id):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

# 5. 增强链 自动附加历史消息
conversation_chain = RunnableWithMessageHistory(
    base_chain,     # 底层基础执行链
    get_history,    # 获取会话ID获取InMemoryChatMessageHistory
    input_messages_key="input",        # 用户输入变量名（模板占位符）
    history_messages_key="chat_history" # 历史对话变量名（模板占位符）
)

if __name__ == '__main__':
    # 固定写法：指定当前会话唯一ID user_001
    session_config = {"configurable": {"session_id": "user_001"}}

    # 三轮连续对话，共用同一会话历史
    print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
    print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
    print(conversation_chain.invoke({"input": "共有几只宠物？"}, session_config))
