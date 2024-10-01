from fasthtml.common import Div, Pre, Code
from fasthtml.components import Img, H2, P
from lucide_fasthtml import Lucide
from shad4fast import Textarea, Button

code_example = """
import datetime
import time

for i in range(10):
    print(f"{datetime.datetime.now()}")
    time.sleep(1)
"""


def SearchBox():
    return Div(
        Div(
            Lucide(icon="search", cls="absolute left-2 top-2 text-muted-foreground"),
            Textarea(
                placeholder="Enter your search query...",
                cls="min-h-12 max-h-[377px] pl-10 pr-14 border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent text-base resize-y overflow-hidden appearance-none",
            ),
            Button(Lucide(icon="arrow-right", size="21"), size="sm", cls="absolute right-2 top-2"),
            cls="relative"
        ),
        cls="grid items-center p-3 bg-muted/80 dark:bg-muted/40 w-full",
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


def Search():
    return Div(
        Div(
            SearchBox(),
            SearchResult(),
            cls="grid"
        ),
        cls="grid",
    )
