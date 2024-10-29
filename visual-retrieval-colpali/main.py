import asyncio
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import uuid
import google.generativeai as genai
from fasthtml.common import (
    Div,
    Img,
    Main,
    P,
    Script,
    Link,
    fast_app,
    HighlightJS,
    FileResponse,
    RedirectResponse,
    Aside,
    StreamingResponse,
    JSONResponse,
    serve,
)
from shad4fast import ShadHead
from vespa.application import Vespa
import base64
from fastcore.parallel import threaded
from PIL import Image

from backend.colpali import get_query_embeddings_and_token_map, gen_similarity_maps
from backend.modelmanager import ModelManager
from backend.vespa_app import VespaQueryClient
from frontend.app import (
    ChatResult,
    Home,
    Search,
    SearchBox,
    SearchResult,
    SimMapButtonPoll,
    SimMapButtonReady,
    WhatIsThis,
)
from frontend.layout import Layout

highlight_js_theme_link = Link(id="highlight-theme", rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(
    langs=["python", "javascript", "java", "json", "xml"],
    dark="github-dark",
    light="github",
)

overlayscrollbars_link = Link(
    rel="stylesheet",
    href="https://cdnjs.cloudflare.com/ajax/libs/overlayscrollbars/2.10.0/styles/overlayscrollbars.min.css",
    type="text/css",
)
overlayscrollbars_js = Script(
    src="https://cdnjs.cloudflare.com/ajax/libs/overlayscrollbars/2.10.0/browser/overlayscrollbars.browser.es5.min.js"
)
awesomplete_link = Link(
    rel="stylesheet",
    href="https://cdnjs.cloudflare.com/ajax/libs/awesomplete/1.1.7/awesomplete.min.css",
    type="text/css",
)
awesomplete_js = Script(
    src="https://cdnjs.cloudflare.com/ajax/libs/awesomplete/1.1.7/awesomplete.min.js"
)
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")

app, rt = fast_app(
    htmlkw={"cls": "grid h-full"},
    pico=False,
    hdrs=(
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme,
        overlayscrollbars_link,
        overlayscrollbars_js,
        awesomplete_link,
        awesomplete_js,
        sselink,
        ShadHead(tw_cdn=False, theme_handle=True),
    ),
)
vespa_app: Vespa = VespaQueryClient()
thread_pool = ThreadPoolExecutor()
# Gemini config

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_SYSTEM_PROMPT = """If the user query is a question, try your best to answer it based on the provided images. 
If the user query can not be interpreted as a question, or if the answer to the query can not be inferred from the images,
answer with the exact phrase "I am sorry, I do not have enough information in the image to answer your question.".
Your response should be HTML formatted, but only simple tags, such as <b>. <p>, <i>, <br> <ul> and <li> are allowed. No HTML tables.
This means that newlines will be replaced with <br> tags, bold text will be enclosed in <b> tags, and so on.
But, you should NOT include backticks (`) or HTML tags in your response.
"""
gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash-8b", system_instruction=GEMINI_SYSTEM_PROMPT
)
STATIC_DIR = Path("static")
IMG_DIR = STATIC_DIR / "full_images"
SIM_MAP_DIR = STATIC_DIR / "sim_maps"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SIM_MAP_DIR, exist_ok=True)


@app.on_event("startup")
def load_model_on_startup():
    app.manager = ModelManager.get_instance()
    return


@app.on_event("startup")
async def keepalive():
    asyncio.create_task(poll_vespa_keepalive())
    return


def generate_query_id(query, ranking_value):
    hash_input = (query + ranking_value).encode("utf-8")
    return hash(hash_input)


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(STATIC_DIR / filepath)


@rt("/")
def get(session):
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return Layout(Main(Home()))


@rt("/what-is-this")
def get():
    return Layout(Main(WhatIsThis()))


@rt("/search")
def get(request):
    # Extract the 'query' and 'ranking' parameters from the URL
    query_value = request.query_params.get("query", "").strip()
    ranking_value = request.query_params.get("ranking", "nn+colpali")
    print("/search: Fetching results for ranking_value:", ranking_value)

    # Always render the SearchBox first
    if not query_value:
        # Show SearchBox and a message for missing query
        return Layout(
            Main(
                Div(
                    SearchBox(query_value=query_value, ranking_value=ranking_value),
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
        )
    # Generate a unique query_id based on the query and ranking value
    query_id = generate_query_id(query_value, ranking_value)
    # Show the loading message if a query is provided
    return Layout(
        Main(Search(request), data_overlayscrollbars_initialize=True, cls="border-t"),
        Aside(
            ChatResult(query_id=query_id, query=query_value),
            cls="border-t border-l hidden md:block",
        ),
    )  # Show SearchBox and Loading message initially


@rt("/fetch_results2")
def get(query: str, ranking: str):
    # 1. Get the results from Vespa (without sim_maps and full_images)
    # Call search-endpoint in Vespa sync.

    # 2. Kick off tasks to fetch sim_maps and full_images
    # Sim maps - call search endpoint async.
    # (A) New rank_profile that does not calculate sim_maps.
    # (A) Make vespa endpoints take select_fields as a parameter.
    # One sim map per image per token.
    # the filename query_id_result_idx_token_idx.png
    # Full image. based on the doc_id.
    # Each of these tasks saves to disk.
    # Need a cleanup task to delete old files.
    # Polling endpoints for sim_maps and full_images checks if file exists and returns it.
    pass


@rt("/fetch_results")
async def get(session, request, query: str, ranking: str):
    if "hx-request" not in request.headers:
        return RedirectResponse("/search")

    # Get the hash of the query and ranking value
    query_id = generate_query_id(query, ranking)
    print(f"Query id in /fetch_results: {query_id}")
    # Run the embedding and query against Vespa app
    model = app.manager.model
    processor = app.manager.processor
    q_embs, idx_to_token = get_query_embeddings_and_token_map(processor, model, query)

    start = time.perf_counter()
    # Fetch real search results from Vespa
    result = await vespa_app.get_result_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
    )
    end = time.perf_counter()
    print(
        f"Search results fetched in {end - start:.2f} seconds, Vespa says searchtime was {result['timing']['searchtime']} seconds"
    )
    search_results = vespa_app.results_to_search_results(result, idx_to_token)
    get_and_store_sim_maps(
        query_id=query_id,
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
    )
    return SearchResult(search_results, query_id)


def get_results_children(result):
    search_results = (
        result["root"]["children"]
        if "root" in result and "children" in result["root"]
        else []
    )
    return search_results


async def poll_vespa_keepalive():
    while True:
        await asyncio.sleep(5)
        await vespa_app.keepalive()
        print(f"Vespa keepalive: {time.time()}")


@threaded
def get_and_store_sim_maps(query_id, query: str, q_embs, ranking, idx_to_token):
    ranking_sim = ranking + "_sim"
    vespa_sim_maps = vespa_app.get_sim_maps_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking_sim,
        idx_to_token=idx_to_token,
    )
    img_paths = [
        IMG_DIR / f"{query_id}_{idx}.jpg" for idx in range(len(vespa_sim_maps))
    ]
    # All images should be downloaded, but best to wait 5 secs
    max_wait = 5
    start_time = time.time()
    while (
        not all([os.path.exists(img_path) for img_path in img_paths])
        and time.time() - start_time < max_wait
    ):
        time.sleep(0.2)
    if not all([os.path.exists(img_path) for img_path in img_paths]):
        print(f"Images not ready in 5 seconds for query_id: {query_id}")
        return False
    sim_map_generator = gen_similarity_maps(
        model=app.manager.model,
        processor=app.manager.processor,
        device=app.manager.device,
        query=query,
        query_embs=q_embs,
        token_idx_map=idx_to_token,
        images=img_paths,
        vespa_sim_maps=vespa_sim_maps,
    )
    for idx, token, token_idx, blended_img_base64 in sim_map_generator:
        with open(SIM_MAP_DIR / f"{query_id}_{idx}_{token_idx}.png", "wb") as f:
            f.write(base64.b64decode(blended_img_base64))
        print(
            f"Sim map saved to disk for query_id: {query_id}, idx: {idx}, token: {token}"
        )
    return True


@app.get("/get_sim_map")
async def get_sim_map(query_id: str, idx: int, token: str, token_idx: int):
    """
    Endpoint that each of the sim map button polls to get the sim map image
    when it is ready. If it is not ready, returns a SimMapButtonPoll, that
    continues to poll every 1 second.
    """
    sim_map_path = SIM_MAP_DIR / f"{query_id}_{idx}_{token_idx}.png"
    if not os.path.exists(sim_map_path):
        print(f"Sim map not ready for query_id: {query_id}, idx: {idx}, token: {token}")
        return SimMapButtonPoll(
            query_id=query_id, idx=idx, token=token, token_idx=token_idx
        )
    else:
        return SimMapButtonReady(
            query_id=query_id,
            idx=idx,
            token=token,
            token_idx=token_idx,
            img_src=sim_map_path,
        )


@app.get("/full_image")
async def full_image(docid: str, query_id: str, idx: int):
    """
    Endpoint to get the full quality image for a given result id.
    """
    img_path = IMG_DIR / f"{query_id}_{idx}.jpg"
    if not os.path.exists(img_path):
        image_data = await vespa_app.get_full_image_from_vespa(docid)
        # image data is base 64 encoded string. Save it to disk as jpg.
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        print(f"Full image saved to disk for query_id: {query_id}, idx: {idx}")
    else:
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    return Img(
        src=f"data:image/jpeg;base64,{image_data}",
        alt="something",
        cls="result-image w-full h-full object-contain",
    )


@rt("/suggestions")
async def get_suggestions(request):
    query = request.query_params.get("query", "").lower().strip()

    if query:
        suggestions = await vespa_app.get_suggestions(query)
        if len(suggestions) > 0:
            return JSONResponse({"suggestions": suggestions})

    return JSONResponse({"suggestions": []})


async def message_generator(query_id: str, query: str):
    images = []
    num_images = 3  # Number of images before firing chat request
    max_wait = 10  # seconds
    start_time = time.time()
    # Check if full images are ready on disk
    while len(images) < num_images and time.time() - start_time < max_wait:
        for idx in range(num_images):
            if not os.path.exists(IMG_DIR / f"{query_id}_{idx}.jpg"):
                print(
                    f"Message generator: Full image not ready for query_id: {query_id}, idx: {idx}"
                )
                continue
            else:
                print(
                    f"Message generator: image ready for query_id: {query_id}, idx: {idx}"
                )
                images.append(Image.open(IMG_DIR / f"{query_id}_{idx}.jpg"))
        await asyncio.sleep(0.2)
    # yield message with number of images ready
    yield f"event: message\ndata: Generating response based on {len(images)} images.\n\n"
    if not images:
        yield "event: message\ndata: I am sorry, I do not have enough information in the image to answer your question.\n\n"
        yield "event: close\ndata: \n\n"
        return

    # If newlines are present in the response, the connection will be closed.
    def replace_newline_with_br(text):
        return text.replace("\n", "<br>")

    response_text = ""
    async for chunk in await gemini_model.generate_content_async(
        images + ["\n\n Query: ", query], stream=True
    ):
        if chunk.text:
            response_text += chunk.text
            response_text = replace_newline_with_br(response_text)
            yield f"event: message\ndata: {response_text}\n\n"
            await asyncio.sleep(0.1)
    yield "event: close\ndata: \n\n"


@app.get("/get-message")
async def get_message(query_id: str, query: str):
    return StreamingResponse(
        message_generator(query_id=query_id, query=query),
        media_type="text/event-stream",
    )


@rt("/app")
def get():
    return Layout(Main(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4")))


if __name__ == "__main__":
    # ModelManager.get_instance()  # Initialize once at startup
    HOT_RELOAD = os.getenv("HOT_RELOAD", "False").lower() == "true"
    print(f"Starting app with hot reload: {HOT_RELOAD}")
    serve(port=7860, reload=HOT_RELOAD)
