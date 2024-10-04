from urllib.parse import quote_plus

from fasthtml.components import Div, H1, P, Img, H2, Form
from lucide_fasthtml import Lucide
from shad4fast import Button, Input


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
    query_value = request.query_params.get('query', '').strip()  # Default to an empty string if not present

    return Div(
        Div(
            SearchBox(query_value=query_value),  # Pass the query value to pre-fill the SearchBox
            SearchResult(),
            cls="grid"
        ),
        cls="grid",
    )


def SearchResult():
    return Div(
        Div(
            Div(
                Img(src='/static/img/sustainability.png', alt='Sustainability',
                    cls='max-w-full h-auto'),
                cls="bg-background px-3 py-5"
            ),
            Div(
                Div(
                    H2("Conocophillips - 2023 Sustainability Report", cls="text-xl font-semibold"),
                    P("Our policies require nature-related risks be assessed in business planning. We disclose our approach to governance, strategy, management and performance related to nature.",
                      cls="text-muted-foreground"),
                    cls="text-sm grid gap-y-4"
                ),
                cls="bg-background px-3 py-5"
            ),
            cls="grid grid-cols-subgrid col-span-2 "
        ),
        cls="grid grid-cols-2 gap-px bg-border"
    )
