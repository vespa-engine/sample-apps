from langgraph.graph import MessagesState
from typing import List

class SubgraphState(MessagesState):
    """State object for the subgraph operations."""

    SearchEngineQuery: str
    ClarifyingQuestion: str
    SearchEngineResults: str
    Filters: List[str]
    Categories: List[str]

    def __init__(self, messages=None, SearchEngineQuery="", ClarifyingQuestion="", SearchEngineResults="", Filters=None, Categories=None):
        """
        Initializes a new SubgraphState instance.

        Args:
            messages (list): List of messages (AI or Human).
            SearchEngineQuery (str): The refined query to pass to the search engine.
            ClarifyingQuestion (str): Any clarification question if the query is ambiguous.
            SearchEngineResults (str): The JSON string of search results.
            Filters (list[str]): List of applied filters like price and rating.
            Categories (list[str]): The identified categories of the product.
        """
        super().__init__(messages=messages or [])
        self.SearchEngineQuery = SearchEngineQuery
        self.ClarifyingQuestion = ClarifyingQuestion
        self.SearchEngineResults = SearchEngineResults
        self.Filters = Filters or []
        self.Categories = Categories or []
