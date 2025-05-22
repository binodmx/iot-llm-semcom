from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import tools_condition, ToolNode
import dotenv
import os

dotenv.load_dotenv(os.getcwd() + "/.env")


def get_llm(model_name):
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name, temperature=0.1)
    elif model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    elif model_name.startswith("claude"):
        return ChatAnthropic(model=model_name, temperature=0.1)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


class Validator:
    def __init__(self, name, model_name, validator_func):
        tools = [validator_func]
        llm = get_llm(model_name)
        llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
        builder = StateGraph(MessagesState)
        builder.add_node(f"{name}",
                         lambda state: {"messages": state["messages"] + [llm_with_tools.invoke(state["messages"])]})
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, f"{name}")
        builder.add_conditional_edges(f"{name}", tools_condition)
        builder.add_edge("tools", f"{name}")
        self.graph = builder.compile()

    def invoke(self, state):
        return self.graph.invoke(state)

    def get_graph(self):
        return self.graph.get_graph()
