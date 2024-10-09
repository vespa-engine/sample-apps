from urllib.parse import quote_plus

from fasthtml.components import H1, H2, Div, Form, Img, P, Span
from fasthtml.xtend import A, Script
from lucide_fasthtml import Lucide
from shad4fast import Badge, Button, Input

# JavaScript to check the input value and enable/disable the search button
check_input_script = Script(
    """
        window.onload = function() {
            const input = document.getElementById('search-input');
            const button = document.querySelector('[data-button="search-button"]');
            function checkInputValue() { button.disabled = input.value.trim() === ""; }
            input.addEventListener('input', checkInputValue);
            checkInputValue();
        };
        """
)

# JavaScript to handle the image swapping and reset button on the search results
image_swapping = Script(
    """
    document.addEventListener('click', function (e) {
        if (e.target.classList.contains('sim-map-button') || e.target.classList.contains('reset-button')) {
            const newSrc = e.target.getAttribute('data-image-src');
            const img = e.target.closest('.relative').querySelector('.result-image');
            img.src = newSrc;
        }
    });
    """
)


def SearchBox(with_border=False, query_value=""):
    grid_cls = "grid gap-2 items-center p-3 bg-muted/80 dark:bg-muted/40 w-full"

    if with_border:
        grid_cls = "grid gap-2 p-3 rounded-md border border-input bg-muted/80 dark:bg-muted/40 w-full ring-offset-background focus-within:outline-none focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:border-input"

    return Form(
        Div(
            Lucide(icon="search", cls="absolute left-2 top-2 text-muted-foreground"),
            Input(
                placeholder="Enter your search query...",
                name="query",
                value=query_value,
                id="search-input",
                cls="text-base pl-10 border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent",
                style="font-size: 1rem",
                autofocus=True,
            ),
            cls="relative",
        ),
        Div(
            Span("controls", cls="text-muted-foreground"),
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
        action=f"/search?query={quote_plus(query_value)}",  # This takes the user to /search with the loading message
        method="GET",
        hx_get=f"/fetch_results?query={quote_plus(query_value)}",  # This fetches the results asynchronously
        hx_trigger="load",  # Trigger this after the page loads
        hx_target="#search-results",  # Update the search results div dynamically
        hx_swap="outerHTML",  # Replace the search results div entirely
        hx_indicator="#loading-indicator",  # Show the loading indicator while fetching results
        cls=grid_cls,
    )


def SampleQueries():
    sample_queries = [
        "Percentage of non-fresh water as source?",
        "Policies related to nature risk?",
        "How much of produced water is recycled?",
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
            cls="text-5xl md:text-7xl font-bold tracking-wide md:tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-black to-gray-700 dark:from-white dark:to-gray-300 animate-fade-in",
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
            cls="grid gap-8 -mt-[34vh]",
        ),
        cls="grid w-full h-full max-w-screen-md items-center gap-4 mx-auto",
    )


def Search(request, search_results=[]):
    query_value = request.query_params.get("query", "").strip()

    return Div(
        Div(
            SearchBox(
                query_value=query_value
            ),  # Pass the query value to pre-fill the SearchBox
            Div(
                LoadingMessage(),  # Show the loading message initially
                id="search-results",  # This will be replaced by the search results
            ),
            cls="grid",
        ),
        cls="grid",
    )


def LoadingMessage():
    return Div(
        P("Loading... Please wait.", cls="text-base text-center"),
        cls="p-10 text-center text-muted-foreground",
        id="loading-indicator",
    )


def SearchResult(results=[], show_sim_map=False):
    if not results:
        return Div(
            P(
                "No results found for your query.",
                cls="text-muted-foreground text-base text-center",
            ),
            cls="grid p-10",
        )

    # Otherwise, display the search results
    result_items = []
    for result in results:
        fields = result["fields"]  # Extract the 'fields' part of each result
        full_image_base64 = f"data:image/jpeg;base64,{fields['full_image']}"

        # Filter sim_map fields that are words with 4 or more characters
        sim_map_fields = {
            key: value
            for key, value in fields.items()
            if key.startswith("sim_map_") and len(key.split("_")[-1]) >= 4
        }

        # Generate buttons for the sim_map fields
        sim_map_buttons = []
        for key, value in sim_map_fields.items():
            sim_map_base64 = f"data:image/jpeg;base64,{value}"
            sim_map_buttons.append(
                Button(
                    key.split("_")[-1],
                    size="sm",
                    data_image_src=sim_map_base64,
                    cls="sim-map-button pointer-events-auto font-mono text-xs h-5 rounded-none px-2",
                )
            )

        # Add "Reset Image" button to restore the full image
        reset_button = Button(
            "Reset",
            variant="outline",
            size="sm",
            data_image_src=full_image_base64,
            cls="reset-button pointer-events-auto font-mono text-xs h-5 rounded-none px-2",
        )

        # Add "Tokens" button - this has no action, just a placeholder
        tokens_button = Button(
            Lucide(icon="images", size="15"),
            "Tokens",
            size="sm",
            cls="bg-[#61D790] text-[#2E2F27] flex gap-[3px] font-bold pointer-events-none font-mono text-xs h-5 rounded-none px-2",
        )

        result_items.append(
            Div(
                Div(
                    Div(
                        tokens_button,
                        *sim_map_buttons,
                        reset_button,
                        cls="flex flex-wrap gap-px w-full  pointer-events-none",
                    ),
                    Img(
                        src=full_image_base64,
                        alt=fields["title"],
                        cls="result-image max-w-full h-auto",
                    ),
                    cls="relative grid gap-px content-start bg-background px-3 py-5",
                ),
                Div(
                    Div(
                        H2(fields["title"], cls="text-xl font-semibold"),
                        P(fields["text"], cls="text-muted-foreground"),
                        cls="text-sm grid gap-y-4",
                    ),
                    cls="bg-background px-3 py-5 hidden md:block",
                ),
                cls="grid grid-cols-1 md:grid-cols-2 col-span-2",
            )
        )

    return Div(
        *result_items,
        image_swapping,
        id="search-results",
        cls="grid grid-cols-2 gap-px bg-border",
    )
