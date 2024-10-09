import asyncio
import uuid

from fasthtml.common import *
from shad4fast import *
from vespa.application import Vespa

from backend.colpali import (
    load_model,
    get_result_from_query,
    get_query_embeddings_and_token_map,
)
from backend.vespa_app import get_vespa_app
from frontend.app import Home, Search, SearchBox, SearchResult
from frontend.layout import Layout

highlight_js_theme_link = Link(id="highlight-theme", rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(
    langs=["python", "javascript", "java", "json", "xml"],
    dark="github-dark",
    light="github",
)
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")


app, rt = fast_app(
    htmlkw={"cls": "h-full"},
    pico=False,
    hdrs=(
        ShadHead(tw_cdn=False, theme_handle=True),
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme,
        sselink,
    ),
)
vespa_app: Vespa = get_vespa_app()

# In-memory storage for the data associated with similarity maps
sim_map_data = {}
# In-memory storage for events to signal when similarity maps are ready
sim_map_events = {}


class ModelManager:
    _instance = None
    model = None
    processor = None

    @staticmethod
    def get_instance():
        if ModelManager._instance is None:
            ModelManager._instance = ModelManager()
            ModelManager._instance.initialize_model_and_processor()
        return ModelManager._instance

    def initialize_model_and_processor(self):
        if self.model is None or self.processor is None:  # Ensure no reinitialization
            self.model, self.processor = load_model()
            if self.model is None or self.processor is None:
                print("Failed to initialize model or processor at startup")
            else:
                print("Model and processor loaded at startup")


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(f"./static/{filepath}")


@rt("/")
def get():
    return Layout(Home())


@rt("/search")
def get(request):
    # Extract the 'query' parameter from the URL using query_params
    query_value = request.query_params.get("query", "").strip()

    # Always render the SearchBox first
    if not query_value:
        # Show SearchBox and a message for missing query
        return Layout(
            Div(
                SearchBox(query_value=query_value),
                Div(
                    P(
                        "No query provided. Please enter a query.",
                        cls="text-center text-muted-foreground",
                    ),
                    cls="p-10",
                ),
                cls="grid",
            )
        )

    # Show the loading message if a query is provided
    return Layout(Search(request))  # Show SearchBox and Loading message initially


@rt("/fetch_results")
async def get(request, query: str, nn: bool = True):
    # Check if the request came from HTMX; if not, redirect to /search
    if "hx-request" not in request.headers:
        return RedirectResponse("/search")

    # Fetch model and processor
    manager = ModelManager.get_instance()
    model = manager.model
    processor = manager.processor
    q_embs, token_to_idx = get_query_embeddings_and_token_map(processor, model, query)
    # Generate a unique identifier for this request
    map_id = str(uuid.uuid4())
    # Create an asyncio Event for this map_id
    sim_map_events[map_id] = asyncio.Event()
    # Start generating the similarity map in the background
    asyncio.create_task(generate_similarity_map(map_id, query))
    # Fetch real search results from Vespa
    result = await get_result_from_query(
        vespa_app,
        processor=processor,
        model=model,
        query=query,
        q_embs=q_embs,
        token_to_idx=token_to_idx,
        nn=nn,
    )
    # Extract search results from the result payload
    search_results = (
        result["root"]["children"]
        if "root" in result and "children" in result["root"]
        else []
    )
    # Directly return the search results without the full page layout
    return SearchResult(search_results, map_id)


# Async function to generate and store the similarity map
async def generate_similarity_map(map_id, query):
    # Simulate a slow calculation for generating the similarity map (e.g., taking 2 seconds)
    await asyncio.sleep(7)
    # manager = ModelManager.get_instance()
    # model = manager.model
    # processor = manager.processor
    # sim_map_result = add_sim_maps_to_result(
    #     result=result,
    #     model=model,
    #     processor=processor,
    #     query=query,
    #     q_embs=q_embs,
    #     token_to_idx=token_to_idx,
    # )
    # Simulate generating the similarity map data
    similarity_map = (
        f"SimilarityMapData for query '{query}'"  # Replace with actual calculation
    )
    # Store the similarity map data on the server associated with the map_id as a Div
    sim_map_data[map_id] = {"query": query, "similarity_map": similarity_map}
    # Signal that the similarity map is ready by setting the event
    sim_map_events[map_id].set()


# Async generator to yield the similarity map via SSE
async def similarity_map_generator(map_id):
    # Retrieve the event associated with the map_id
    event = sim_map_events.get(map_id)
    if event is None:
        # If event not found, inform the client and close the connection
        yield "event: error\ndata: Similarity map not found.\n\n"
        yield "event: close\ndata: \n\n"
        return
    # Wait until the similarity map is ready
    await event.wait()
    # Similarity map should now be ready; retrieve it
    data = sim_map_data.get(map_id)
    if data is None:
        # Data not found even after event is set
        yield "event: error\ndata: Similarity map not found after event.\n\n"
        yield "event: close\ndata: \n\n"
        return
    # Prepare the similarity map content using the stored data
    similarity_map_content = (
        f"Similarity Map for query '{data['query']}': {data['similarity_map']}"
    )
    # Yield the similarity map via SSE
    yield f"event: update\ndata: {similarity_map_content}\n\n"
    # Signal that the SSE stream can be closed
    yield "event: close\ndata: \n\n"
    # Clean up the stored data and event
    del sim_map_data[map_id]
    del sim_map_events[map_id]


# SSE endpoint to stream the similarity map, accepting the map_id parameter
@app.get("/similarity-map")
async def similarity_map(map_id: str):
    return StreamingResponse(
        similarity_map_generator(map_id), media_type="text/event-stream"
    )


@rt("/app")
def get():
    return Layout(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4"))


if __name__ == "__main__":
    # ModelManager.get_instance()  # Initialize once at startup
    serve(port=7860)
