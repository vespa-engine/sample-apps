from fasthtml.components import Div, H1, P, Span, A
from lucide_fasthtml import Lucide
from shad4fast import Button, Textarea


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


def SearchBox():
    return Div(
        Div(
            Lucide(icon="search", cls="absolute left-2 top-2 text-muted-foreground"),
            Textarea(
                placeholder="Enter your search query...",
                cls="max-h-[377px] pl-10 border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent text-base resize-y overflow-hidden appearance-none",
            ),
            cls="relative"
        ),
        Div(
            Span(
                "tests",
                cls="text-muted-foreground"
            ),
            Div(
                A(Button(Lucide(icon="arrow-right", size="21"), size="sm"), href="/search"),
            ),
            cls="flex justify-between"
        ),
        cls="grid gap-2 p-3 rounded-md border border-input bg-muted/80 dark:bg-muted/40 w-full ring-offset-background focus-within:outline-none focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:border-input",
    )


def Home():
    return Div(
        Div(
            Hero(),
            SearchBox(),
            cls="grid gap-8 -mt-[34vh]"
        ),
        cls="grid w-full h-full max-w-screen-md items-center gap-4 mx-auto"
    )
