import os
from langgraph.graph import StateGraph
from models.state import SubgraphState
from langgraph_modules.subgraph import subgraph
from langchain_openai import ChatOpenAI
from langgraph.graph import START
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import tools_condition, ToolNode
from utils.secrets import OPENAI_API_KEY, TAVILY_API_KEY

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not os.environ.get("TAVILY_API_KEY"):  
  os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def VespaRetrieverTool(UserQuery: str) -> str:
    """Retrieves all items for sale matching the user query.

    Args:
        UserQuery: User Query for items they are looking for.
    """

    # Initiate a conversation with the UserQuery as a human message
    conversation = [ HumanMessage(content=UserQuery) ]

    # Create an instance of SubgraphState with default values.
    # Make sure the constructor of SubgraphState accepts the required parameters.
    subgraph_state = SubgraphState(
        messages=conversation,
        SearchEngineQuery="",
        ClarifyingQuestion="",
        SearchEngineResults="",
        Filters=['quantity > 0'],
        Categories=[]
    )

    # Test Invocation
    thread_config = {"configurable": {"thread_id": "some_id"}}

    # Invoke graph with a valid SubgraphState object
    result_state=subgraph.invoke(subgraph_state, config=thread_config)

    #Fetch final result
    result=result_state["SearchEngineResults"]

    return result

TavilySearch = TavilySearchResults(max_results=2)

tools = [TavilySearch, VespaRetrieverTool]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""You are a helpful sales assistant willing to answer any user questions about items to sell. You will abide by the following rules:
                                   - Rule 1: You will try your best to provide all the information regarding an item for sale to a customer.
                                   - Rule 2: When the user asks you a question unrelated to an item for sale, you can try to address the question, but always prompt the user back to ask for any items for sale, or relate their question to an item they might be looking for.
                                   - Rule 3: You may get a clarifying question as a result. In this case, please ask the question to the user. You do not need to be apologetic.
                                   - Rule 4: You may get an empty resultset as an answer. In this case, apologize and ask the the user if they are looking for something else.
                                    """)
# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
