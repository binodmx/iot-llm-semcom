from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
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


class Codec:
    def __init__(self, name, model_name):
        llm = get_llm(model_name)
        builder = StateGraph(MessagesState)
        builder.add_node(f"{name}", lambda state: {"messages": state["messages"] + [llm.invoke(state["messages"])]})
        builder.add_edge(START, f"{name}")
        builder.add_edge(f"{name}", END)
        self.graph = builder.compile()

    def invoke(self, state):
        return self.graph.invoke(state)

    def get_graph(self):
        return self.graph.get_graph()
