from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from models.state import SubgraphState
from vespa_module.vespa_retriever import VespaRetriever
from utils.helpers import get_latest_ai_or_human_message
from langchain_openai import ChatOpenAI
import streamlit as st
from typing import Literal
from langgraph.graph import START, END



def subgraph_GenerateKeywords (state: SubgraphState):

    conversation=state["messages"]
   
    userquery = get_latest_ai_or_human_message("HumanMessage",state)

    categories=state["Categories"]

    response_schemas = [
     ResponseSchema(name="SearchEngineQuestion", description="Actual keywords to pass to the Search Engine.")
        ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get the format instructions
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""Considering the context of the conversation, the last user query, and the current categories to which the item belongs to, your role is to generate search terms as specific as possible to search
                    for items in the backend Vespa search engine. You should abide by the following rules:
                            Rule 1: At least 2 search terms are necessary.
                            Rule 2: The words are related directly to the item description in the conversation. 
                            Rule 3: Do not make up attributes not specified by the user. 
                            Rule 4: It shouldn't have any words describing its price or rating.
                            Rule 5: Apply stemming and lemmatization if required.  
                    Format Instructions: {format_instructions}
                    Conversation Context: {conversation}
                    User Query: {userquery}
                    Categories: {categories}""",
        input_variables=["conversation", "userquery","categories"],
        partial_variables={"format_instructions": format_instructions},
    )

    # Initialize the language model
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    #model = ChatOllama(model="phi4")  

    # Create the chain
    chain = prompt | model | output_parser

    # Invoke the chain
    llm_response = chain.invoke({"conversation": conversation, "userquery": userquery, "categories": categories})

    state["SearchEngineQuery"] = llm_response.get("SearchEngineQuestion")

    return state

def subgraph_EvaluateCategory (state: SubgraphState):

    conversation=state["messages"]
        
    userquery = get_latest_ai_or_human_message("HumanMessage",state)
    
    response_schemas = [
    ResponseSchema(name="score", description="Binary score with value 'yes' or 'no'. Respond yes if you have identified the potential categories to which that item belongs in the list. Respond no if you need to further confirm with the user"),
    ResponseSchema(name="Categories", description="If the score is yes, return a category or list of category from the list in the format of a string array. For example: ['All_Beauty']"),
    ResponseSchema(name="ClarifyingQuestion", description="If the score is no, ask a question with the top 3 to 5 potential categories the item could belong to from the list provided. If the score is yes, return the string 'None'")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get the format instructions
    format_instructions = output_parser.get_format_instructions()


    # Create the prompt template
    prompt = PromptTemplate(
        template="""You are evaluating the user inquiry on behalf of a sales assistant agent and assess if the query contains enough information to determine concisely one category to which the item belongs.
        The categories to choose from are the following:
	    1.	Industrial_and_Scientific: Professional-grade tools, lab equipment, industrial machinery, and safety supplies.
	    2.	Kindle_Store: Digital books, magazines, and audiobooks available for Kindle e-readers and apps.
	    3.	Musical_Instruments: Guitars, keyboards, drums, and accessories for musicians, including beginner and professional gear.
	    4.	Office_Products: Office supplies, stationery, printers, desks, and organizers for home and business use.
	    5.	Patio_Lawn_and_Garden: Outdoor furniture, gardening tools, plants, BBQ grills, and landscaping accessories.
	    6.	Pet_Supplies: Food, toys, grooming tools, and healthcare products for pets including dogs, cats, and birds.
	    7.	Sports_and_Outdoors: Equipment, apparel, and accessories for fitness, sports, and outdoor activities like hiking and camping.
	    8.	Tools_and_Home_Improvement: Hardware tools, power equipment, plumbing, electrical, and DIY home repair essentials.
	    9.	Toys_and_Games: A variety of toys, puzzles, board games, and collectibles for kids and adults.
	    \n{format_instructions}\n,
        Conversation Context: {conversation}\n"
        User Query: {userquery}""",
        input_variables=["conversation", "userquery"],
        partial_variables={"format_instructions": format_instructions},
    )

    # Initialize the language model
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    #model = ChatOllama(model="phi4") 
    
    # Create the chain
    chain = prompt | model | output_parser

    # Invoke the chain
    llm_response = chain.invoke({"conversation": conversation,"userquery": userquery})  
    #print(llm_response)

    # Populate the state based on LLM response
    if llm_response.get("score") == "yes":
        state["Categories"] = llm_response.get("Categories")
        state["ClarifyingQuestion"] = "None"  # No need to clarify further
    else:
        state["Categories"] = ['None']  # No valid search query
        state["ClarifyingQuestion"] = llm_response.get("ClarifyingQuestion", "")

    return state

def subgraph_EvaluateFilters (state: SubgraphState):

    conversation=state["messages"]
    userquery = get_latest_ai_or_human_message("HumanMessage",state)
    
    response_schemas = [
    ResponseSchema(name="Filters", description="""Returns an array of strings delimited by double quotes corresponding to filters to pass to the search engine.
                                                  Based on the schema of the query, filters can be on the following fields in the data schema:
                                                        - Field price: price representing the price of the item. The filter can have price > or < or = or >= or <= than a numerical integer value. Example "price <= 200"
                                                        - Field average_rating: average_rating represents the rating of the item. The average_rating could be > or < or = or >= or <= than a numerical float value. Example: "average_rating > 4.5"
                                                  If no filters are identified, return an array of string with an entry 'quantity > 0'
                                                """)
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get the format instructions
    format_instructions = output_parser.get_format_instructions()


    # Create the prompt template
    prompt = PromptTemplate(
        template="""Your role is to identify any filters associated to a price or a rating from the current conversation.
                    {format_instructions}\n,
                    Conversation Context: {conversation}\n
                    User Query: {userquery}""",
        input_variables=["conversation", "userquery"],
        partial_variables={"format_instructions": format_instructions},
    )

    # Initialize the language model
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    #model = ChatOllama(model="phi4")
    
    # Create the chain
    chain = prompt | model | output_parser

    # Invoke the chain
    llm_response = chain.invoke({"conversation": conversation, "userquery": userquery})  

    # Populate the state based on LLM response
    state["Filters"] = llm_response.get("Filters", ['quantity > 0'])
    
    # Check if "quantity > 0" is NOT in the list
    if "quantity > 0" not in state["Filters"]:
        state["Filters"].append("quantity > 0")

    return state

def EvaluateResults(state: SubgraphState):
    
    SearchEngineResults=state["SearchEngineResults"]
    conversation=state["messages"]
    userquery=state["SearchEngineQuery"]
    

    response_schemas = [
    ResponseSchema(name="FinalResults", description="""Returns an array of strings delimited by double quotes corresponding to the final subset of results to return to the user. The result should include the title, the description, the price, the average rating.
                                                """)
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # Create the prompt template
    prompt = PromptTemplate(
 	template="""You are a result evaluator and your role is to evaluate the search results and ensure result accuracy. You will follow the following the steps to evaluate the result set:
                    - Step 1: Eliminate any result that is irrelevant for the user query in the search results. If the item has no relevance to the user query, it should be removed from the result set.
                    - Step 2: Assign a relevancy score between 0 and 100 to each remaining result. The higher the score, the more relevant the result.
                    - Step 3: Order the resulting set from the previous step by decreasing order of relevancy
                    - Step 4: Truncate and return up to the first five relevant results, eliminating any non-relevant results.
                    - Step 5: If the resulting list is empty, you should return the string 'None'.
                    {format_instructions}\n,
                    Conversation Context: {conversation}\n
                    User Query: {userquery}\n
                    Search Engine Results: {searchengineresults}""",
        input_variables=["conversation", "userquery","searchengineresults"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    # Initialize the language model
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    #model = ChatOllama(model="phi4")
    
    # Create the chain
    chain = prompt | model | output_parser

    # Invoke the chain
    llm_response = chain.invoke({"conversation": conversation, "userquery": userquery, "searchengineresults": SearchEngineResults})  

    # Populate the state based on LLM response
    state["SearchEngineResults"] = llm_response.get("FinalResults", ['None'])
    print(state["SearchEngineResults"])
    return state

def human_node(state: SubgraphState):
    print("Hello from human_node")
    ClarifyingQuestion=state["ClarifyingQuestion"]
    state["messages"].append(AIMessage(content=ClarifyingQuestion))
    print(ClarifyingQuestion)

    # Populate the state based on LLM response
    state["SearchEngineResults"] = "Sorry can't answer the question. Could you please clarify the following with the user: " + ClarifyingQuestion

    return state

def CheckforClarificationCategory(state: SubgraphState)-> Literal["GetHumanFeedback", "GenerateQueryTerms"]:
    
    CQ = state["ClarifyingQuestion"]
    
    if CQ == 'None':
            return "GenerateQueryTerms"
    else: return "GetHumanFeedback"


# Define the subgraph
subgraph_builder = StateGraph(SubgraphState)

# Add nodes
subgraph_builder.add_node("GenerateMainPredicate", subgraph_EvaluateCategory)
subgraph_builder.add_node("GenerateQueryTerms", subgraph_GenerateKeywords)
subgraph_builder.add_node("GetHumanFeedback", human_node)
subgraph_builder.add_node("RunVespaRetriever", VespaRetriever)
subgraph_builder.add_node("GenerateFilters", subgraph_EvaluateFilters)
subgraph_builder.add_node("EvaluateResults", EvaluateResults) 

# Define edges
subgraph_builder.add_edge(START, "GenerateMainPredicate")  
#subgraph_builder.add_edge("GetHumanFeedback", "GenerateMainPredicate") 
subgraph_builder.add_edge("GenerateQueryTerms", "GenerateFilters")
subgraph_builder.add_edge("GenerateFilters","RunVespaRetriever")
subgraph_builder.add_edge("RunVespaRetriever", "EvaluateResults")
subgraph_builder.add_edge("GetHumanFeedback", END)

# Add conditional Edge
subgraph_builder.add_conditional_edges("GenerateMainPredicate", CheckforClarificationCategory)

# Compile
subgraph = subgraph_builder.compile()
