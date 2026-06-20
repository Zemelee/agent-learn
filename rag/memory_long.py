import json
import os
from typing import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import messages_from_dict, message_to_dict, BaseMessage
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str = "./chat"):
        self.session_id = session_id
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    @property # 调用时不必带括号
    def messages(self) -> list[BaseMessage]:
        try:
            with open(
                os.path.join(self.storage_path, self.session_id),
                "r",
                encoding="utf-8",
            ) as f:
                messages_data = json.load(f)
            return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)  # Existing messages
        all_messages.extend(messages)  # Add new messages
        serialized = [message_to_dict(message) for message in all_messages]
        file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(self.storage_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False)

    def clear(self) -> None:
        file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(self.storage_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


chat = ChatTongyi(model="qwen3.6-max-preview")


def print_prompt(full_prompt):
    print("=" * 20, full_prompt.to_string(), "=" * 20)
    return full_prompt


prompt = PromptTemplate.from_template(
    "你需要根据对话历史回应用户问题。对话历史:{chat_history}。用户当前输入: {input}，请回答"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据对话历史回应用户问题。"),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答如下问题：{input}"),
    ]
)


# 3. 基础执行链：模板 -> 打印调试prompt -> 模型调用 -> 字符串解析
base_chain = prompt | print_prompt | chat | StrOutputParser()


# 4. 全局会话存储字典：key=session_id，value=InMemoryChatMessageHistory实例
chat_history_store = {}


# 根据session_id获取对应会话历史，不存在则新建内存会话
def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat")


# 5. 增强链 自动附加历史消息
conversation_chain = RunnableWithMessageHistory(
    base_chain,  # 底层基础执行链
    get_history,  # 获取会话ID获取InMemoryChatMessageHistory
    input_messages_key="input",  # 用户输入变量名（模板占位符）
    history_messages_key="chat_history",  # 历史对话变量名（模板占位符）
)

if __name__ == "__main__":
    # 固定写法：指定当前会话唯一ID user_001
    session_config = {"configurable": {"session_id": "user_001"}}

    # 三轮连续对话，共用同一会话历史
    # print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
    # print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
    print(conversation_chain.invoke({"input": "共有几只宠物？"}, session_config))
