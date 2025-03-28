import streamlit as st
from PIL import Image
from langgraph_modules.main_graph import builder
from langgraph_modules.subgraph import subgraph_builder
from langchain_core.messages import AIMessage


# Streamlit UI
st.title(":shopping_trolley: Vespa.ai Shopping Assistant")

# Load and display sidebar image
icon = Image.open("Vespa-logo-dark-RGB.png")
st.sidebar.image(icon, width=500)

subgraph = subgraph_builder.compile()
react_graph = builder.compile()

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I'm your Vespa Shopping Assistant using an agentic architecture based on LangGraph. How can I assist you today ?",
        }
    ]

# Prompt the user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from the assistant, we need to generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    message_list = []
    messages = react_graph.invoke(
        {"messages": st.session_state.messages}, stream_mode="values"
    )

    for m in messages["messages"]:
        message_list.append(m)

    response_text = next(
        (
            msg.content
            for msg in reversed(messages["messages"])
            if isinstance(msg, AIMessage)
        ),
        None,
    )

    if response_text:
        # Save assistant response before displaying
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        # Display assistant's response in chat
        with st.chat_message("assistant"):
            st.write(response_text)
