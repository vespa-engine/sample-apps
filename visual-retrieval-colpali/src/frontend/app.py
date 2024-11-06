from typing import Optional
from urllib.parse import quote_plus

from fasthtml.components import H1, H2, H3, Br, Div, Form, Img, NotStr, P, Span
from fasthtml.xtend import A, Script
from lucide_fasthtml import Lucide
from shad4fast import Badge, Button, Input, Label, RadioGroup, RadioGroupItem, Separator

# JavaScript to check the input value and enable/disable the search button and radio buttons
check_input_script = Script(
    """
        window.onload = function() {
            const input = document.getElementById('search-input');
            const button = document.querySelector('[data-button="search-button"]');
            const radioGroupItems = document.querySelectorAll('button[data-ref="radio-item"]');  // Get all radio buttons
            
            function checkInputValue() {
                const isInputEmpty = input.value.trim() === "";
                button.disabled = isInputEmpty;  // Disable the submit button
                radioGroupItems.forEach(item => {
                    item.disabled = isInputEmpty;  // Disable/enable the radio buttons
                });
            }

            input.addEventListener('input', checkInputValue);  // Listen for input changes
            checkInputValue();  // Initial check when the page loads
        };
    """
)

# JavaScript to handle the image swapping, reset button, and active class toggling
image_swapping = Script(
    """
    document.addEventListener('click', function (e) {
        if (e.target.classList.contains('sim-map-button') || e.target.classList.contains('reset-button')) {
            const imgContainer = e.target.closest('.relative'); 
            const overlayContainer = imgContainer.querySelector('.overlay-container');
            const newSrc = e.target.getAttribute('data-image-src');
    
            // If it's a reset button, remove the overlay image
            if (e.target.classList.contains('reset-button')) {
                overlayContainer.innerHTML = '';  // Clear the overlay container, showing only the full image
            } else {
                // Create a new overlay image
                const img = document.createElement('img');
                img.src = newSrc;
                img.classList.add('overlay-image', 'absolute', 'top-0', 'left-0', 'w-full', 'h-full');
                overlayContainer.innerHTML = '';  // Clear any previous overlay
                overlayContainer.appendChild(img);  // Add the new overlay image
            }
    
            // Toggle active class on buttons
            const activeButton = document.querySelector('.sim-map-button.active');
            if (activeButton) {
                activeButton.classList.remove('active');
            }
            if (e.target.classList.contains('sim-map-button')) {
                e.target.classList.add('active');
            }
        }
    });
    """
)

toggle_text_content = Script(
    """
    function toggleTextContent(idx) {
        const textColumn = document.getElementById(`text-column-${idx}`);
        const imageTextColumns = document.getElementById(`image-text-columns-${idx}`);
        const toggleButton = document.getElementById(`toggle-button-${idx}`);
    
        if (textColumn.classList.contains('md-grid-text-column')) {
          // Hide the text column
          textColumn.classList.remove('md-grid-text-column');
          imageTextColumns.classList.remove('grid-image-text-columns');
          toggleButton.innerText = `Show Text`;
        } else {
          // Show the text column
          textColumn.classList.add('md-grid-text-column');
          imageTextColumns.classList.add('grid-image-text-columns');
          toggleButton.innerText = `Hide Text`;
        }
    }
    """
)

autocomplete_script = Script(
    """
    document.addEventListener('DOMContentLoaded', function() {
        const input = document.querySelector('#search-input');
        const awesomplete = new Awesomplete(input, { minChars: 1, maxItems: 5 });

        input.addEventListener('input', function() {
            if (this.value.length >= 1) {
                // Use template literals to insert the input value dynamically in the query parameter
                fetch(`/suggestions?query=${encodeURIComponent(this.value)}`)
                    .then(response => response.json())
                    .then(data => {
                        // Update the Awesomplete list dynamically with fetched suggestions
                        awesomplete.list = data.suggestions;
                    })
                    .catch(err => console.error('Error fetching suggestions:', err));
            }
        });
    });
    """
)

dynamic_elements_scrollbars = Script(
    """
    (function () {
        const { applyOverlayScrollbars, getScrollbarTheme } = OverlayScrollbarsManager;

        function applyScrollbarsToDynamicElements() {
            const scrollbarTheme = getScrollbarTheme();

            // Apply scrollbars to dynamically loaded result-text-full and result-text-snippet elements
            const resultTextFullElements = document.querySelectorAll('[id^="result-text-full"]');
            const resultTextSnippetElements = document.querySelectorAll('[id^="result-text-snippet"]');

            resultTextFullElements.forEach(element => {
                applyOverlayScrollbars(element, scrollbarTheme);
            });

            resultTextSnippetElements.forEach(element => {
                applyOverlayScrollbars(element, scrollbarTheme);
            });
        }

        // Apply scrollbars after dynamic content is loaded (e.g., after search results)
        applyScrollbarsToDynamicElements();

        // Observe changes in the 'dark' class to adjust the theme dynamically if needed
        const observer = new MutationObserver(applyScrollbarsToDynamicElements);
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    })();
    """
)


def SearchBox(with_border=False, query_value="", ranking_value="nn+colpali"):
    grid_cls = "grid gap-2 items-center p-3 bg-muted w-full"

    if with_border:
        grid_cls = "grid gap-2 p-3 rounded-md border border-input bg-muted w-full ring-offset-background focus-within:outline-none focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:border-input"

    return Form(
        Div(
            Lucide(
                icon="search", cls="absolute left-2 top-2 text-muted-foreground z-10"
            ),
            Input(
                placeholder="Enter your search query...",
                name="query",
                value=query_value,
                id="search-input",
                cls="text-base pl-10 border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent bg-white dark:bg-background awesomplete",
                data_list="#suggestions",
                style="font-size: 1rem",
                autofocus=True,
            ),
            cls="relative",
        ),
        Div(
            Div(
                Span("Ranking by:", cls="text-muted-foreground text-xs font-semibold"),
                RadioGroup(
                    Div(
                        RadioGroupItem(value="nn+colpali", id="nn+colpali"),
                        Label("ColPali", htmlFor="ColPali"),
                        cls="flex items-center space-x-2",
                    ),
                    Div(
                        RadioGroupItem(value="bm25", id="bm25"),
                        Label("BM25", htmlFor="BM25"),
                        cls="flex items-center space-x-2",
                    ),
                    Div(
                        RadioGroupItem(value="bm25+colpali", id="bm25+colpali"),
                        Label("Hybrid ColPali + BM25", htmlFor="Hybrid ColPali + BM25"),
                        cls="flex items-center space-x-2",
                    ),
                    name="ranking",
                    default_value=ranking_value,
                    cls="grid-flow-col gap-x-5 text-muted-foreground",
                ),
                cls="grid grid-flow-col items-center gap-x-3 border border-input px-3 rounded-sm",
            ),
            Button(
                Lucide(icon="arrow-right", size="21"),
                size="sm",
                type="submit",
                data_button="search-button",
                disabled=True,
            ),
            cls="flex justify-between",
        ),
        check_input_script,
        autocomplete_script,
        action=f"/search?query={quote_plus(query_value)}&ranking={quote_plus(ranking_value)}",
        method="GET",
        hx_get="/fetch_results",  # As the component is a form, input components query and ranking are sent as query parameters automatically, see https://htmx.org/docs/#parameters
        hx_trigger="load",
        hx_target="#search-results",
        hx_swap="outerHTML",
        hx_indicator="#loading-indicator",
        cls=grid_cls,
    )


def SampleQueries():
    sample_queries = [
        "Total amount of fixed salaries paid in 2023?",
        "Proportion of female new hires 2021-2023?",
        "Number of internship applications trend 2021-2023",
        "Gender balance at level 4 or above in NY office 2023?",
        "What percentage of the funds unlisted real estate investments were in Switzerland 2023?",
        "child jumping over puddle",
        "hula hoop kid",
    ]

    query_badges = []
    for query in sample_queries:
        query_badges.append(
            A(
                Badge(
                    Div(
                        Lucide(
                            icon="text-search", size="18", cls="text-muted-foreground"
                        ),
                        Span(query, cls="text-base font-normal"),
                        cls="flex gap-2 items-center",
                    ),
                    variant="outline",
                    cls="text-base font-normal text-muted-foreground hover:border-black dark:hover:border-white",
                ),
                href=f"/search?query={quote_plus(query)}",
                cls="no-underline",
            )
        )

    return Div(*query_badges, cls="grid gap-2 justify-items-center")


def Hero():
    return Div(
        H1(
            "Vespa.ai + ColPali",
            cls="text-5xl md:text-7xl font-bold tracking-wide md:tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-black to-slate-700 dark:from-white dark:to-slate-300 animate-fade-in",
        ),
        P(
            "Efficient Document Retrieval with Vision Language Models",
            cls="text-lg md:text-2xl text-muted-foreground md:tracking-wide",
        ),
        cls="grid gap-5 text-center",
    )


def Home():
    return Div(
        Div(
            Hero(),
            SearchBox(with_border=True),
            SampleQueries(),
            cls="grid gap-8 content-start mt-[13vh]",
        ),
        cls="grid w-full h-full max-w-screen-md gap-4 mx-auto",
    )


def AboutThisDemo():
    return Div(
        Div(
            Div(
                H1(
                    "Vespa.ai + ColPali",
                    cls="text-5xl font-bold tracking-wide md:tracking-wider",
                ),
                P(
                    "Efficient Document Retrieval with Vision Language Models",
                    cls="text-lg text-muted-foreground md:tracking-wide",
                ),
                Div(
                    Img(
                        src="/static/img/vespa-colpali.png",
                        alt="Vespa and ColPali",
                        cls="object-contain h-[377px]",
                    ),
                    cls="grid justify-center",
                ),
                Div(
                    P(
                        "This is a demo application showcasing the integration of Vespa.ai and ColPali for visual retrieval of documents.",
                        cls="text-base",
                    ),
                    P(
                        "The application uses a combination of neural networks and traditional search algorithms to retrieve relevant documents based on visual and textual queries.",
                        cls="text-base",
                    ),
                    cls="grid gap-2 text-center",
                ),
                cls="grid gap-5 text-center",
            ),
            cls="grid gap-8 content-start mt-[8vh]",
        ),
        cls="grid w-full h-full max-w-screen-md gap-4 mx-auto",
    )


def Search(request, search_results=[]):
    query_value = request.query_params.get("query", "").strip()
    ranking_value = request.query_params.get("ranking", "nn+colpali")
    print(
        f"Search: Fetching results for query: {query_value}, ranking: {ranking_value}"
    )
    return Div(
        Div(
            Div(
                SearchBox(query_value=query_value, ranking_value=ranking_value),
                Div(
                    LoadingMessage(),
                    id="search-results",  # This will be replaced by the search results
                ),
                cls="grid",
            ),
            cls="grid",
        ),
    )


def LoadingMessage(display_text="Retrieving search results"):
    return Div(
        Lucide(icon="loader-circle", cls="size-5 mr-1.5 animate-spin"),
        Span(display_text, cls="text-base text-center"),
        cls="p-10 text-muted-foreground flex items-center justify-center",
        id="loading-indicator",
    )


def LoadingSkeleton():
    return Div(
        Div(cls="h-5 bg-muted"),
        Div(cls="h-5 bg-muted"),
        Div(cls="h-5 bg-muted"),
        cls="grid gap-2 animate-pulse",
    )


def SimMapButtonReady(query_id, idx, token, token_idx, img_src):
    return Button(
        token.replace("\u2581", ""),
        size="sm",
        data_image_src=img_src,
        id=f"sim-map-button-{query_id}-{idx}-{token_idx}-{token}",
        cls="sim-map-button pointer-events-auto font-mono text-xs h-5 rounded-none px-2",
    )


def SimMapButtonPoll(query_id, idx, token, token_idx):
    return Button(
        Lucide(icon="loader-circle", size="15", cls="animate-spin"),
        size="sm",
        disabled=True,
        hx_get=f"/get_sim_map?query_id={query_id}&idx={idx}&token={token}&token_idx={token_idx}",
        hx_trigger="every 0.5s",
        hx_swap="outerHTML",
        cls="pointer-events-auto text-xs h-5 rounded-none px-2",
    )


def SearchInfo(search_time, total_count):
    return (
        Div(
            NotStr(
                f"<span>Found <strong>{total_count}</strong> results in <strong>{search_time}</strong> seconds.</span>"
            ),
            cls="grid bg-background border-t text-sm text-center p-3",
        ),
    )


def SearchResult(
    results: list,
    query: str,
    query_id: Optional[str] = None,
    search_time: float = 0,
    total_count: int = 0,
):
    if not results:
        return Div(
            P(
                "No results found for your query.",
                cls="text-muted-foreground text-base text-center",
            ),
            cls="grid p-10",
        )

    doc_ids = []
    # Otherwise, display the search results
    result_items = []
    for idx, result in enumerate(results):
        fields = result["fields"]  # Extract the 'fields' part of each result
        doc_id = fields["id"]
        doc_ids.append(doc_id)
        blur_image_base64 = f"data:image/jpeg;base64,{fields['blur_image']}"

        sim_map_fields = {
            key: value
            for key, value in fields.items()
            if key.startswith(
                "sim_map_"
            )  # filtering is done before creating with 'should_filter_token'-function
        }

        # Generate buttons for the sim_map fields
        sim_map_buttons = []
        for key, value in sim_map_fields.items():
            token = key.split("_")[-2]
            token_idx = int(key.split("_")[-1])
            if value is not None:
                sim_map_base64 = f"data:image/jpeg;base64,{value}"
                sim_map_buttons.append(
                    SimMapButtonReady(
                        query_id=query_id,
                        idx=idx,
                        token=token,
                        token_idx=token_idx,
                        img_src=sim_map_base64,
                    )
                )
            else:
                sim_map_buttons.append(
                    SimMapButtonPoll(
                        query_id=query_id,
                        idx=idx,
                        token=token,
                        token_idx=token_idx,
                    )
                )

        # Add "Reset Image" button to restore the full image
        reset_button = Button(
            "Reset",
            variant="outline",
            size="sm",
            data_image_src=blur_image_base64,
            cls="reset-button pointer-events-auto font-mono text-xs h-5 rounded-none px-2",
        )

        tokens_icon = Lucide(icon="images", size="15")

        # Add "Tokens" button - this has no action, just a placeholder
        tokens_button = Button(
            tokens_icon,
            "Tokens",
            size="sm",
            cls="tokens-button flex gap-[3px] font-bold pointer-events-none font-mono text-xs h-5 rounded-none px-2",
        )

        result_items.append(
            Div(
                Div(
                    Div(
                        Lucide(icon="file-text"),
                        H2(fields["title"], cls="text-xl md:text-2xl font-semibold"),
                        Separator(orientation="vertical"),
                        Badge(
                            f"Relevance score: {result['relevance']:.4f}",
                            cls="flex gap-1.5 items-center justify-center",
                        ),
                        cls="flex items-center gap-2",
                    ),
                    Div(
                        Button(
                            "Hide Text",
                            size="sm",
                            id=f"toggle-button-{idx}",
                            onclick=f"toggleTextContent({idx})",
                            cls="hidden md:block",
                        ),
                    ),
                    cls="flex flex-wrap items-center justify-between bg-background px-3 py-4",
                ),
                Div(
                    Div(
                        Div(
                            tokens_button,
                            *sim_map_buttons,
                            reset_button,
                            cls="flex flex-wrap gap-px w-full pointer-events-none",
                        ),
                        Div(
                            Div(
                                Div(
                                    Img(
                                        src=blur_image_base64,
                                        hx_get=f"/full_image?doc_id={doc_id}",
                                        style="backdrop-filter: blur(5px);",
                                        hx_trigger="load",
                                        hx_swap="outerHTML",
                                        alt=fields["title"],
                                        cls="result-image w-full h-full object-contain",
                                    ),
                                    Div(
                                        cls="overlay-container absolute top-0 left-0 w-full h-full pointer-events-none"
                                    ),
                                    cls="relative w-full h-full",
                                ),
                                cls="grid bg-muted p-2",
                            ),
                            cls="block",
                        ),
                        id=f"image-column-{idx}",
                        cls="image-column relative bg-background px-3 py-5 grid-image-column",
                    ),
                    Div(
                        Div(
                            A(
                                Lucide(icon="external-link", size="18"),
                                f"PDF Source (Page {fields['page_number']})",
                                href=f"{fields['url']}#page={fields['page_number'] + 1}",
                                target="_blank",
                                cls="flex items-center gap-1.5 font-mono bold text-sm",
                            ),
                            cls="flex items-center justify-end",
                        ),
                        Div(
                            Div(
                                Div(
                                    Div(
                                        Div(
                                            H3(
                                                "Dynamic summary",
                                                cls="text-base font-semibold",
                                            ),
                                            P(
                                                NotStr(fields.get("snippet", "")),
                                                cls="text-highlight text-muted-foreground",
                                            ),
                                            cls="grid grid-rows-[auto_0px] content-start gap-y-3",
                                        ),
                                        id=f"result-text-snippet-{idx}",
                                        cls="grid gap-y-3 p-8 border border-dashed",
                                    ),
                                    Div(
                                        Div(
                                            Div(
                                                H3(
                                                    "Full text",
                                                    cls="text-base font-semibold",
                                                ),
                                                Div(
                                                    P(
                                                        NotStr(fields.get("text", "")),
                                                        cls="text-highlight text-muted-foreground",
                                                    ),
                                                    Br(),
                                                ),
                                                cls="grid grid-rows-[auto_0px] content-start gap-y-3",
                                            ),
                                            id=f"result-text-full-{idx}",
                                            cls="grid gap-y-3 p-8 border border-dashed",
                                        ),
                                        Div(
                                            cls="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#fcfcfd] dark:from-[#1c2024] pt-[7%]"
                                        ),
                                        cls="relative grid",
                                    ),
                                    cls="grid grid-rows-[1fr_1fr] xl:grid-rows-[1fr_2fr] gap-y-8 p-8 text-sm",
                                ),
                                cls="grid bg-background",
                            ),
                            cls="grid bg-muted p-2",
                        ),
                        id=f"text-column-{idx}",
                        cls="text-column relative bg-background px-3 py-5 hidden md-grid-text-column",
                    ),
                    id=f"image-text-columns-{idx}",
                    cls="relative grid grid-cols-1 border-t grid-image-text-columns",
                ),
                cls="grid grid-cols-1 grid-rows-[auto_auto_1fr]",
            ),
        )

    return [
        Div(
            SearchInfo(search_time, total_count),
            *result_items,
            image_swapping,
            toggle_text_content,
            dynamic_elements_scrollbars,
            id="search-results",
            cls="grid grid-cols-1 gap-px bg-border min-h-0",
        ),
        Div(
            ChatResult(query_id=query_id, query=query, doc_ids=doc_ids),
            hx_swap_oob="true",
            id="chat_messages",
        ),
    ]


def ChatResult(query_id: str, query: str, doc_ids: Optional[list] = None):
    messages = Div(LoadingSkeleton())

    if doc_ids:
        messages = Div(
            LoadingSkeleton(),
            hx_ext="sse",
            sse_connect=f"/get-message?query_id={query_id}&doc_ids={','.join(doc_ids)}&query={quote_plus(query)}",
            sse_swap="message",
            sse_close="close",
            hx_swap="innerHTML",
        )

    return Div(
        Div("AI-response (Gemini-8B)", cls="text-xl font-semibold p-5"),
        Div(
            Div(
                messages,
            ),
            id="chat-messages",
            cls="overflow-auto min-h-0 grid items-end px-5",
        ),
        id="chat_messages",
        cls="h-full grid grid-rows-[auto_1fr_auto] min-h-0 gap-3",
    )
