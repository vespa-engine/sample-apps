import asyncio
import base64
import os
import time
import uuid
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import google.generativeai as genai
from fastcore.parallel import threaded
from fasthtml.common import (
    Aside,
    Div,
    FileResponse,
    HighlightJS,
    Img,
    JSONResponse,
    Link,
    Main,
    P,
    RedirectResponse,
    Script,
    StreamingResponse,
    fast_app,
    serve,
)
from PIL import Image
from shad4fast import ShadHead
from vespa.application import Vespa

from backend.colpali import SimMapGenerator
from backend.vespa_app import VespaQueryClient
from frontend.app import (
    AboutThisDemo,
    ChatResult,
    Home,
    Search,
    SearchBox,
    SearchResult,
    SimMapButtonPoll,
    SimMapButtonReady,
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

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Configure logger
logger = logging.getLogger("vespa_app")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(levelname)s: \t %(asctime)s \t %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL))

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
vespa_app: Vespa = VespaQueryClient(logger=logger)
thread_pool = ThreadPoolExecutor()
# Gemini config

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_SYSTEM_PROMPT = """If the user query is a question, try your best to answer it based on the provided images. 
If the user query can not be interpreted as a question, or if the answer to the query can not be inferred from the images,
answer with the exact phrase "I am sorry, I can't find enough relevant information on these pages to answer your question.".
Your response should be HTML formatted, but only simple tags, such as <b>. <p>, <i>, <br> <ul> and <li> are allowed. No HTML tables.
This means that newlines will be replaced with <br> tags, bold text will be enclosed in <b> tags, and so on.
Do NOT include backticks (`) in your response. Only simple HTML tags and text.
"""
gemini_model = genai.GenerativeModel(
    "gemini-2.0-flash", system_instruction=GEMINI_SYSTEM_PROMPT
)
STATIC_DIR = Path("static")
IMG_DIR = STATIC_DIR / "full_images"
SIM_MAP_DIR = STATIC_DIR / "sim_maps"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SIM_MAP_DIR, exist_ok=True)


@app.on_event("startup")
def load_model_on_startup():
    app.sim_map_generator = SimMapGenerator(logger=logger)
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
    return Layout(Main(Home()), is_home=True)


@rt("/about-this-demo")
def get():
    return Layout(Main(AboutThisDemo()))


@rt("/search")
def get(request, query: str = "", ranking: str = "hybrid"):
    logger.info(f"/search: Fetching results for query: {query}, ranking: {ranking}")

    # Always render the SearchBox first
    if not query:
        # Show SearchBox and a message for missing query
        return Layout(
            Main(
                Div(
                    SearchBox(query_value=query, ranking_value=ranking),
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
    query_id = generate_query_id(query, ranking)
    # Show the loading message if a query is provided
    return Layout(
        Main(Search(request), data_overlayscrollbars_initialize=True, cls="border-t"),
        Aside(
            ChatResult(query_id=query_id, query=query),
            cls="border-t border-l hidden md:block",
        ),
    )  # Show SearchBox and Loading message initially


@rt("/fetch_results")
async def get(session, request, query: str, ranking: str):
    if "hx-request" not in request.headers:
        return RedirectResponse("/search")

    # Get the hash of the query and ranking value
    query_id = generate_query_id(query, ranking)
    logger.info(f"Query id in /fetch_results: {query_id}")
    # Run the embedding and query against Vespa app
    start_inference = time.perf_counter()
    q_embs, idx_to_token = app.sim_map_generator.get_query_embeddings_and_token_map(
        query
    )
    end_inference = time.perf_counter()
    logger.info(
        f"Inference time for query_id: {query_id} \t {end_inference - start_inference:.2f} seconds"
    )

    start = time.perf_counter()
    # Fetch real search results from Vespa
    result = await vespa_app.get_result_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
    )
    end = time.perf_counter()
    logger.info(
        f"Search results fetched in {end - start:.2f} seconds. Vespa search time: {result['timing']['searchtime']}"
    )
    search_time = result["timing"]["searchtime"]
    # Safely get total_count with a default of 0
    total_count = result.get("root", {}).get("fields", {}).get("totalCount", 0)

    search_results = vespa_app.results_to_search_results(result, idx_to_token)

    get_and_store_sim_maps(
        query_id=query_id,
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
        doc_ids=[result["fields"]["id"] for result in search_results],
    )
    return SearchResult(search_results, query, query_id, search_time, total_count)


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
        logger.debug(f"Vespa keepalive: {time.time()}")


@threaded
def get_and_store_sim_maps(
    query_id, query: str, q_embs, ranking, idx_to_token, doc_ids
):
    ranking_sim = ranking + "_sim"
    vespa_sim_maps = vespa_app.get_sim_maps_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking_sim,
        idx_to_token=idx_to_token,
    )
    img_paths = [IMG_DIR / f"{doc_id}.jpg" for doc_id in doc_ids]
    # All images should be downloaded, but best to wait 5 secs
    max_wait = 5
    start_time = time.time()
    while (
        not all([os.path.exists(img_path) for img_path in img_paths])
        and time.time() - start_time < max_wait
    ):
        time.sleep(0.2)
    if not all([os.path.exists(img_path) for img_path in img_paths]):
        logger.warning(f"Images not ready in 5 seconds for query_id: {query_id}")
        return False
    sim_map_generator = app.sim_map_generator.gen_similarity_maps(
        query=query,
        query_embs=q_embs,
        token_idx_map=idx_to_token,
        images=img_paths,
        vespa_sim_maps=vespa_sim_maps,
    )
    for idx, token, token_idx, blended_img_base64 in sim_map_generator:
        with open(SIM_MAP_DIR / f"{query_id}_{idx}_{token_idx}.png", "wb") as f:
            f.write(base64.b64decode(blended_img_base64))
        logger.debug(
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
        logger.debug(
            f"Sim map not ready for query_id: {query_id}, idx: {idx}, token: {token}"
        )
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
async def full_image(doc_id: str):
    """
    Endpoint to get the full quality image for a given result id.
    """
    img_path = IMG_DIR / f"{doc_id}.jpg"
    if not os.path.exists(img_path):
        image_data = await vespa_app.get_full_image_from_vespa(doc_id)
        # image data is base 64 encoded string. Save it to disk as jpg.
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        logger.debug(f"Full image saved to disk for doc_id: {doc_id}")
    else:
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    return Img(
        src=f"data:image/jpeg;base64,{image_data}",
        alt="something",
        cls="result-image w-full h-full object-contain",
    )


@rt("/suggestions")
async def get_suggestions(query: str = ""):
    """Endpoint to get suggestions as user types in the search box"""
    query = query.lower().strip()

    if query:
        suggestions = await vespa_app.get_suggestions(query)
        if len(suggestions) > 0:
            return JSONResponse({"suggestions": suggestions})

    return JSONResponse({"suggestions": []})


async def message_generator(query_id: str, query: str, doc_ids: list):
    """Generator function to yield SSE messages for chat response"""
    images = []
    num_images = 3  # Number of images before firing chat request
    max_wait = 10  # seconds
    start_time = time.time()
    # Check if full images are ready on disk
    while (
        len(images) < min(num_images, len(doc_ids))
        and time.time() - start_time < max_wait
    ):
        images = []
        for idx in range(num_images):
            image_filename = IMG_DIR / f"{doc_ids[idx]}.jpg"
            if not os.path.exists(image_filename):
                logger.debug(
                    f"Message generator: Full image not ready for query_id: {query_id}, idx: {idx}"
                )
                continue
            else:
                logger.debug(
                    f"Message generator: image ready for query_id: {query_id}, idx: {idx}"
                )
                images.append(Image.open(image_filename))
        if len(images) < num_images:
            await asyncio.sleep(0.2)

    # yield message with number of images ready
    yield f"event: message\ndata: Generating response based on {len(images)} images...\n\n"
    if not images:
        yield "event: message\ndata: Failed to send images to Gemini 2.0!\n\n"
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
async def get_message(query_id: str, query: str, doc_ids: str):
    return StreamingResponse(
        message_generator(query_id=query_id, query=query, doc_ids=doc_ids.split(",")),
        media_type="text/event-stream",
    )


@rt("/app")
def get():
    return Layout(Main(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4")))


if __name__ == "__main__":
    HOT_RELOAD = os.getenv("HOT_RELOAD", "False").lower() == "true"
    logger.info(f"Starting app with hot reload: {HOT_RELOAD}")
    serve(port=7860, reload=HOT_RELOAD)
