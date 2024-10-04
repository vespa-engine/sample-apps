from urllib.parse import quote_plus

from fasthtml.components import Div, H1, P, Img, H2, Form, Span
from lucide_fasthtml import Lucide
from shad4fast import Button, Input


def get_mock_results(query):
    # Mock data simulating search results for the query
    mock_data = [
        {
            "title": "Conocophillips - 2023 Sustainability Report",
            "description": "Our policies require nature-related risks be assessed in business planning...",
            "image": "/static/img/sustainability.png"
        },
        {
            "title": "Sustainable Energy in the 21st Century",
            "description": "An overview of sustainable energy practices and future technologies...",
            "image": "/static/img/energy.png"
        },
        {
            "title": "Reducing Carbon Emissions by 2030",
            "description": "Steps we can take to reduce global carbon emissions by the year 2030...",
            "image": "/static/img/carbon.png"
        }
    ]

    # For simulation
    if query:
        return [result for result in mock_data if query.lower() in result['title'].lower()]

    return mock_data


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
                cls="text-base pl-10 border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent",
                style='font-size: 1rem',
                autofocus=True,
            ),
            cls="relative"
        ),
        Div(
            Span(
                "controls",
                cls="text-muted-foreground"
            ),
            Button(Lucide(icon="arrow-right", size="21"), size="sm", type="submit"),
            cls="flex justify-between"
        ),
        action=f"/search?query={quote_plus(query_value)}",
        method="GET",
        cls=grid_cls,
    )


def Hero():
    return Div(
        H1(
            "Vespa.Ai + ColPali",
            cls="text-5xl md:text-7xl font-bold tracking-wide md:tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-black to-gray-700 dark:from-white dark:to-gray-300 animate-fade-in"
        ),
        P(
            "Efficient Document Retrieval with Vision Language Models",
            cls="text-lg md:text-2xl text-muted-foreground md:tracking-wide"
        ),
        cls="grid gap-5 text-center"
    )


def Home():
    return Div(
        Div(
            Hero(),
            SearchBox(with_border=True),
            cls="grid gap-8 -mt-[34vh]"
        ),
        cls="grid w-full h-full max-w-screen-md items-center gap-4 mx-auto"
    )


def Search(request):
    # Extract the 'query' parameter from the URL using query_params
    query_value = request.query_params.get('query', '').strip()

    # Get mock search results based on the query
    search_results = get_mock_results(query_value)

    return Div(
        Div(
            SearchBox(query_value=query_value),  # Pass the query value to pre-fill the SearchBox
            SearchResult(results=search_results),
            cls="grid"
        ),
        cls="grid",
    )


def SearchResult(results=[]):
    result_items = []
    for result in results:
        result_items.append(
            Div(
                Div(
                    Img(src=result['image'], alt=result['title'], cls='max-w-full h-auto'),
                    cls="bg-background px-3 py-5"
                ),
                Div(
                    Div(
                        H2(result['title'], cls="text-xl font-semibold"),
                        P(result['description'], cls="text-muted-foreground"),
                        cls="text-sm grid gap-y-4"
                    ),
                    cls="bg-background px-3 py-5"
                ),
                cls="grid grid-cols-subgrid col-span-2"
            )
        )

    return Div(
        *result_items,
        cls="grid grid-cols-2 gap-px bg-border"
    )
