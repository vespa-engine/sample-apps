import asyncio
import base64
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import google.generativeai as genai
from fasthtml.common import *
from PIL import Image
from shad4fast import *
from vespa.application import Vespa

from backend.cache import LRUCache
from backend.colpali import (
    add_sim_maps_to_result,
    get_query_embeddings_and_token_map,
    is_special_token,
)
from backend.modelmanager import ModelManager
from pathlib import Path
from backend.vespa_app import VespaQueryClient
from frontend.app import (
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
result_cache = LRUCache(max_size=20)  # Each result can be ~10MB
task_cache = LRUCache(
    max_size=1000
)  # Map from query_id to boolean value - False if not all results are ready.
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
STATIC_DIR = Path(__file__).parent / "static"
IMG_DIR = STATIC_DIR / "saved"
os.makedirs(STATIC_DIR, exist_ok=True)


@app.on_event("startup")
def load_model_on_startup():
    app.manager = ModelManager.get_instance()
    return


@app.on_event("startup")
async def keepalive():
    asyncio.create_task(poll_vespa_keepalive())
    return


def generate_query_id(query):
    return hashlib.md5(query.encode("utf-8")).hexdigest()


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(STATIC_DIR / filepath)


@rt("/")
def get():
    return Layout(Main(Home()))


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
    query_id = generate_query_id(query_value + ranking_value)
    # See if results are already in cache
    # if result_cache.get(query_id) is not None:
    #     print(f"Results for query_id {query_id} already in cache")
    #     result = result_cache.get(query_id)
    #     search_results = get_results_children(result)
    #     return Layout(Search(request, search_results))
    # Show the loading message if a query is provided
    return Layout(
        Main(Search(request), data_overlayscrollbars_initialize=True, cls="border-t"),
        Aside(
            ChatResult(query_id=query_id, query=query_value),
            cls="border-t border-l hidden md:block",
        ),
    )  # Show SearchBox and Loading message initially


@rt("/fetch_results")
async def get(request, query: str, nn: bool = True):
    if "hx-request" not in request.headers:
        return RedirectResponse("/search")

    # Extract ranking option from the request
    ranking_value = request.query_params.get("ranking")
    print(
        f"/fetch_results: Fetching results for query: {query}, ranking: {ranking_value}"
    )
    # Generate a unique query_id based on the query and ranking value
    query_id = generate_query_id(query + ranking_value)
    # See if results are already in cache
    # if result_cache.get(query_id) is not None:
    #     print(f"Results for query_id {query_id} already in cache")
    #     result = result_cache.get(query_id)
    #     search_results = get_results_children(result)
    #     return SearchResult(search_results, query_id)
    # Run the embedding and query against Vespa app
    task_cache.set(query_id, False)
    model = app.manager.model
    processor = app.manager.processor
    q_embs, token_to_idx = get_query_embeddings_and_token_map(processor, model, query)

    start = time.perf_counter()
    # Fetch real search results from Vespa
    result = await vespa_app.get_result_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking_value,
        token_to_idx=token_to_idx,
    )
    end = time.perf_counter()
    print(
        f"Search results fetched in {end - start:.2f} seconds, Vespa says searchtime was {result['timing']['searchtime']} seconds"
    )
    # Add result to cache
    result_cache.set(query_id, result)
    # Start generating the similarity map in the background
    asyncio.create_task(
        generate_similarity_map(
            model, processor, query, q_embs, token_to_idx, result, query_id
        )
    )
    fields_to_add = [
        f"sim_map_{token}"
        for token in token_to_idx.keys()
        if not is_special_token(token)
    ]
    search_results = get_results_children(result)
    for result in search_results:
        for sim_map_key in fields_to_add:
            result["fields"][sim_map_key] = None
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


async def generate_similarity_map(
    model, processor, query, q_embs, token_to_idx, result, query_id
):
    loop = asyncio.get_event_loop()
    sim_map_task = partial(
        add_sim_maps_to_result,
        result=result,
        model=model,
        processor=processor,
        query=query,
        q_embs=q_embs,
        token_to_idx=token_to_idx,
        query_id=query_id,
        result_cache=result_cache,
    )
    sim_map_result = await loop.run_in_executor(thread_pool, sim_map_task)
    result_cache.set(query_id, sim_map_result)
    task_cache.set(query_id, True)


@app.get("/get_sim_map")
async def get_sim_map(query_id: str, idx: int, token: str):
    """
    Endpoint that each of the sim map button polls to get the sim map image
    when it is ready. If it is not ready, returns a SimMapButtonPoll, that
    continues to poll every 1 second.
    """
    result = result_cache.get(query_id)
    if result is None:
        return SimMapButtonPoll(query_id=query_id, idx=idx, token=token)
    search_results = get_results_children(result)
    # Check if idx exists in list of children
    if idx >= len(search_results):
        return SimMapButtonPoll(query_id=query_id, idx=idx, token=token)
    else:
        sim_map_key = f"sim_map_{token}"
        sim_map_b64 = search_results[idx]["fields"].get(sim_map_key, None)
        if sim_map_b64 is None:
            return SimMapButtonPoll(query_id=query_id, idx=idx, token=token)
        sim_map_img_src = f"data:image/png;base64,{sim_map_b64}"
        return SimMapButtonReady(
            query_id=query_id, idx=idx, token=token, img_src=sim_map_img_src
        )


async def update_full_image_cache(docid: str, query_id: str, idx: int, image_data: str):
    result = None
    max_wait = 20  # seconds. If horribly slow network latency.
    start_time = time.time()
    while result is None and time.time() - start_time < max_wait:
        result = result_cache.get(query_id)
        if result is None:
            await asyncio.sleep(0.1)
    try:
        result["root"]["children"][idx]["fields"]["full_image"] = image_data
    except KeyError as err:
        print(f"Error updating full image cache: {err}")
    result_cache.set(query_id, result)
    print(f"Full image cache updated for query_id {query_id}")
    return


@app.get("/full_image")
async def full_image(docid: str, query_id: str, idx: int):
    """
    Endpoint to get the full quality image for a given result id.
    """
    image_data = await vespa_app.get_full_image_from_vespa(docid)
    # Update the cache with the full image data
    asyncio.create_task(update_full_image_cache(docid, query_id, idx, image_data))
    return Img(
        src=f"data:image/png;base64,{image_data}",
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
    result = None
    all_images_ready = False
    max_wait = 10  # seconds
    start_time = time.time()
    while not all_images_ready and time.time() - start_time < max_wait:
        result = result_cache.get(query_id)
        if result is None:
            await asyncio.sleep(0.1)
            continue
        search_results = get_results_children(result)
        for single_result in search_results:
            img = single_result["fields"].get("full_image", None)
            if img is not None:
                images.append(img)
                if len(images) == len(search_results):
                    all_images_ready = True
                    break
            else:
                await asyncio.sleep(0.1)

    # from b64 to PIL image
    images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images]
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
    serve(port=7860)
