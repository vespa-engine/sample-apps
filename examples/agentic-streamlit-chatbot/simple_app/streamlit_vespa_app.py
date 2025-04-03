
import streamlit as st
import getpass
import os

from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse
import json

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from PIL import Image

st.title(":shopping_trolley: Vespa.ai Shopping Assistant")

# Load an image (ensure the file is in the correct path)
icon = Image.open("Vespa-logo-dark-RGB.png")

# Display the image in the sidebar
st.sidebar.image(icon, width=500)

# Fetch secrets
OPENAI_API_KEY = st.secrets["api_keys"]["llm"]
TAVILY_API_KEY = st.secrets["api_keys"]["tavily"]
VESPA_URL = st.secrets["vespa"]["url"]
PUBLIC_CERT_PATH = st.secrets["vespa"]["public_cert_path"]
PRIVATE_KEY_PATH = st.secrets["vespa"]["private_key_path"]

#if not os.environ.get("OPENAI_API_KEY"):
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#if not os.environ.get("TAVILY_API_KEY"):
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


def VespaRetriever(UserQuery: str) -> str:
    """Retrieves all items for sale matching the user query.

    Args:
        UserQuery: User Query for items they are looking for.
    """

    vespa_app = Vespa(url=VESPA_URL,
                  cert=PUBLIC_CERT_PATH,
                  key=PRIVATE_KEY_PATH)

    with vespa_app.syncio(connections=1) as session:
        query = UserQuery
        response: VespaQueryResponse = session.query(
            yql="select id, category, title, average_rating, price from sources * where userQuery()",
            query=query,
            hits=5,
            ranking="hybrid",
        )
    assert response.is_successful()

    # Extract only the 'fields' content from each entry
    filtered_data = [hit["fields"] for hit in response.hits]

    # Convert to a JSON string
    json_string = json.dumps(filtered_data, indent=1)

    return json_string

TavilySearch = TavilySearchResults(max_results=2)

tools = [TavilySearch, VespaRetriever]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful sales assistant willing to answer any user questions about items to sell. You will try your best to provide all the information regarding an item for sale to a customer.")

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

react_graph = builder.compile()

#Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
       {"role": "assistant", "content": "Hello, I'm your Vespa Shopping Assistant using an agentic architecture based on LangGraph. How can I assist you today ?"}
    ]

#Prompt the user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

#Display the existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
       st.write(message["content"])

#If last message is not from the assistant, we need to generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
   question = st.session_state.messages[-1]["content"].replace('?','')

   message_list = []
   messages = react_graph.invoke({"messages": st.session_state.messages}, stream_mode="values")
   print(messages)

   for m in messages['messages']:
       message_list.append(m)
    
   print(message_list)

   response_text = next(
    (msg.content for msg in reversed(messages['messages']) if isinstance(msg, AIMessage)),
    None
    )
   with st.chat_message("assistant"):
       st.write("Response: ", response_text)
   
       # **Add the assistant response to session state**
   st.session_state.messages.append({"role": "assistant", "content": response_text})
