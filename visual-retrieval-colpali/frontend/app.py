from fasthtml.components import Div, H1, P, Span, A, Img, H2, Pre, Code
from lucide_fasthtml import Lucide
from shad4fast import Textarea, Button

code_example = """
import datetime
import time

for i in range(10):
    print(f"{datetime.datetime.now()}")
    time.sleep(1)
"""


def SearchBox(with_border=False):
    grid_cls = "grid items-center p-3 bg-muted/80 dark:bg-muted/40 w-full"

    if with_border:
        grid_cls = "grid gap-2 p-3 rounded-md border border-input bg-muted/80 dark:bg-muted/40 w-full ring-offset-background focus-within:outline-none focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:border-input"

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


def Search():
    return Div(
        Div(
            SearchBox(),
            SearchResult(),
            cls="grid"
        ),
        cls="grid",
    )


def SearchResult():
    return Div(
        Div(
            Div(
                Img(src='https://assets.vespa.ai/logos/vespa-logo-black.svg', alt='Vespa Logo Black',
                    cls='max-w-full h-auto'),
                cls="bg-background px-3 py-5"
            ),
            Div(
                Div(
                    H2("Vespa Ai Black Logo", cls="text-xl font-semibold"),
                    P("This is a description for the Vespa AI logo in black. Vespa is a platform for real-time AI and search applications. It provides advanced services for fast, relevant, and scalable search results.",
                      cls="text-muted-foreground"),
                    Pre(Code(code_example)),
                    cls="text-sm grid gap-y-4"
                ),
                cls="bg-background px-3 py-5"
            ),
            cls="grid grid-cols-subgrid col-span-2 "
        ),
        Div(
            Div(
                Img(src='https://assets.vespa.ai/logos/vespa-logo-white.svg', alt='Vespa Logo White',
                    cls='max-w-full h-auto'),
                cls="bg-background px-3 py-5"
            ),
            Div(
                Div(
                    H2("Vespa Ai White Logo", cls="text-xl font-semibold"),
                    P("This is a description for the Vespa AI logo in white. It highlights the adaptability of the brand and its applications across different visual media and backgrounds.",
                      cls="text-muted-foreground"),
                    cls="text-sm grid gap-y-4"
                ),
                cls="bg-background px-3 py-5"
            ),
            cls="grid grid-cols-subgrid col-span-2 "
        ),
        Div(
            Div(
                Img(src='https://assets.vespa.ai/logos/vespa-logo-black.svg', alt='Vespa Logo Black',
                    cls='max-w-full h-auto'),
                cls="bg-background px-3 py-5"
            ),
            Div(
                Div(
                    H2("Another Result for Vespa Logo Black", cls="text-xl font-semibold"),
                    P("This result refers to an alternative context where the black Vespa logo is used. It's commonly seen in dark-themed interfaces.",
                      cls="text-muted-foreground"),
                    cls="text-sm grid gap-y-4"
                ),
                cls="bg-background px-3 py-5"
            ),
            cls="grid grid-cols-subgrid col-span-2 "
        ),
        Div(
            Div(
                Img(src='https://assets.vespa.ai/logos/vespa-logo-white.svg', alt='Vespa Logo White',
                    cls='max-w-full h-auto'),
                cls="bg-background px-3 py-5"
            ),
            Div(
                Div(
                    H2("Another Result for Vespa Logo White", cls="text-xl font-semibold"),
                    P("Here we see another search result referring to the white Vespa logo. This design is perfect for dark backgrounds where the white logo stands out clearly.",
                      cls="text-muted-foreground"),
                    Pre(Code(code_example)),
                    cls="text-sm grid gap-y-4"
                ),
                cls="bg-background px-3 py-5"
            ),
            cls="grid grid-cols-subgrid col-span-2 "
        ),
        cls="grid grid-cols-2 gap-px bg-border"
    )
