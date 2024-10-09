from fasthtml.common import *
import asyncio
from starlette.responses import StreamingResponse
import uuid  # For generating unique identifiers

# Set up the app with Tailwind CSS, DaisyUI, HTMX, and the HTMX SSE extension
tlink = Script(src="https://cdn.tailwindcss.com")
dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)
htmxlink = Script(src="https://unpkg.com/htmx.org@1.9.3")
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
app = FastHTML(hdrs=(tlink, dlink, htmxlink, sselink), live=True)

# In-memory storage for the data associated with similarity maps
sim_map_data = {}
# In-memory storage for events to signal when similarity maps are ready
sim_map_events = {}


# Route to serve static files (if needed)
@app.get("/{fname:path}.{ext:static}")
def static(fname: str, ext: str):
    return FileResponse(f"{fname}.{ext}")


# Async function to generate and store the similarity map
async def generate_similarity_map(map_id, query):
    # Simulate a slow calculation for generating the similarity map (e.g., taking 2 seconds)
    await asyncio.sleep(2)
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
    # Store the similarity map data on the server associated with the map_id
    sim_map_data[map_id] = {"query": query, "similarity_map": similarity_map}
    # Signal that the similarity map is ready by setting the event
    sim_map_events[map_id].set()


# The main page with the search input field and button
@app.get("/")
def index():
    page = Body(
        H1("Search Page"),
        # The search form
        Form(
            Input(
                type="text",
                name="query",
                placeholder="Enter search query",
                cls="input input-bordered w-full",
                required=True,
            ),
            Button("Search", type="submit", cls="btn btn-primary mt-2"),
            action="/search",
            method="post",
            hx_post="/search",  # Use HTMX to make the POST request
            hx_target="#search-results",  # Target the search-results div
            hx_swap="innerHTML",  # Replace the content inside search-results div
            cls="form-control",
        ),
        # Div to display the search results
        Div(id="search-results", cls="mt-4"),
        cls="p-4 max-w-lg mx-auto",
    )
    return Title("Search Page"), page


# Endpoint to handle the POST request when the search button is clicked
@app.post("/search")
async def search(query: str):
    # Generate a unique identifier for this request
    map_id = str(uuid.uuid4())
    # Create an asyncio Event for this map_id
    sim_map_events[map_id] = asyncio.Event()
    # Start generating the similarity map in the background
    asyncio.create_task(generate_similarity_map(map_id, query))
    # Immediately return the fast result
    fast_result = Div(
        P(f"This is the fast result for '{query}'."),
        # Div to receive the similarity map
        Div(
            id="similarity-map",
            # Set up the SSE connection to receive the similarity map
            hx_ext="sse",
            # Include the map_id in the SSE connection URL
            sse_connect=f"/similarity-map?map_id={map_id}",
            sse_swap="update",  # SSE event name to listen for
            sse_close="close",  # Event to signal closing the SSE connection
            hx_swap="innerHTML",  # Swap the content of this Div when similarity map arrives
        ),
    )
    return fast_result


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
    similarity_map_content = f"<p>This is the similarity map for '{data['query']}'.<br>Data: {data['similarity_map']}</p>"
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


serve()
