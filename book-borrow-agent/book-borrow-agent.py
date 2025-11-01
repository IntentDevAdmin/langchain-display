import os
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


# 1、定义系统提示词
SYSTEM_PROMPT = """
您是一位知识渊博的图书管理员，您精通职场心理学，同时也擅长做职场心理疏导，擅长员工激励。
您可使用以下两种工具：
- get_book_list_by_userid：用于获取该员工的借阅书籍列表。
- get_book_list：用于获取可以借阅的书籍列表。
当您通过工具查看到员工之前所借阅的书籍时，您要根据书籍的内容给员工输出一段极具鼓励的话语。并推荐一本未被借阅的书给员工，并说明推荐理由。若员工愿意和你继续聊起你推荐的书籍的话题，您也可以根据您所学到的该书籍的知识和员工畅聊。
"""

# 2、定义上下文结构
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# 3、定义可用工具
@tool
def get_book_list() -> dict:
    """获取可借阅的书籍列表"""
    books = [
        {
            "title": "心理学与生活", 
            "author": "理查德·格里格", 
            "publisher": "人民邮电出版社", 
            "publish_date": "2016-01-01", 
            "description": "本书介绍了心理学的基本概念和理论，帮助读者理解人类行为和心理过程。", 
            "available": True
        },
        {
            "title": "影响力", 
            "author": "罗伯特·西奥迪尼", 
            "publisher": "中国人民大学出版社", 
            "publish_date": "2010-01-01", 
            "description": "本书揭示了影响力的六大原则，帮助读者掌握说服他人的技巧。", 
            "available": True
        },
        {
            "title": "思考，快与慢", 
            "author": "丹尼尔·卡尼曼", 
            "publisher": "中信出版社", 
            "publish_date": "2012-01-01", 
            "description": "本书探讨了人类思维的两种模式，揭示了决策过程中的认知偏差。", 
            "available": False
        },
        {
            "title": "自控力", 
            "author": "凯利·麦格尼格尔", 
            "publisher": "机械工业出版社", 
            "publish_date": "2013-01-01", 
            "description": "本书提供了提升自控力的科学方法，帮助读者克服拖延和冲动。", 
            "available": True
        },
        {
            "title": "心流", 
            "author": "米哈里·契克森米哈赖", 
            "publisher": "浙江人民出版社", 
            "publish_date": "2017-01-01", 
            "description": "本书探讨了心流状态的特征和实现方法，帮助读者提升专注力和幸福感。", 
            "available": True
        }
    ]
    return {"books": [book for book in books if book['available']]}


@tool
def get_book_list_by_userid(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID获取该用户的借阅书籍列表"""
    user_id = runtime.context.user_id
    borrowed_books = {
        "1": ["心理学与生活 - 理查德·格里格", "影响力 - 罗伯特·西奥迪尼"],
        "2": ["思考，快与慢 - 丹尼尔·卡尼曼"]
    }
    return "\n".join(borrowed_books.get(user_id, ["该用户暂无借阅记录"]))


# 4、配置大模型
model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# 5、定义响应格式
@dataclass
class ResponseFormat:
    """智能体的响应结构定义。
    encouragement_response: 鼓励的话语
    book_info: 推荐的书籍信息，Json格式，包含书名、作者、出版社、出版日期、简介等字段。
    """
    # 一段鼓励的话语
    encouragement_response: str
    # 推荐书籍信息，Json格式
    book_info: dict | None=None
    # 推荐理由
    reason: str | None=None

# 6、设置记忆
checkpointer = InMemorySaver()

# 7、创建Agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_book_list, get_book_list_by_userid],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# 8、运行Agent
# thread_id 是特定会话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "查询我的借阅书籍"}]},
    config=config,
    context=Context(user_id="1")
)

print("====== Agent返回的结构化响应 ======")
print(response['structured_response'])
print("====== 解析Agent返回的各个字段 ======")
print("鼓励话语:", response['structured_response'].encouragement_response)
print("推荐书籍:", response['structured_response'].book_info)
print("推荐理由:", response['structured_response'].reason)
