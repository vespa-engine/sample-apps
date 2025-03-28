from langchain_core.messages import AIMessage, HumanMessage

def get_latest_ai_or_human_message(messagetype, state):
    for message in reversed(state["messages"]):
        if (messagetype == 'AIMessage' and isinstance(message, AIMessage)) or \
           (messagetype == 'HumanMessage' and isinstance(message, HumanMessage)):
            return message.content
    return None