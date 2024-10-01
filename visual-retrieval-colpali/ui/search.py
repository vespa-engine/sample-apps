from fasthtml.components import Div
from lucide_fasthtml import Lucide
from shad4fast import Textarea, Button


def SearchBox():
    return Div(
        Div(
            Lucide(icon="search", cls="absolute left-2 top-2 text-muted-foreground"),
            Textarea(
                placeholder="Enter your search query...",
                cls="min-h-[55px] max-h-[377px] pl-10 pr-14 border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent text-base resize-y overflow-hidden appearance-none",
            ),
            Button(Lucide(icon="arrow-right", size="21"), size="sm", cls="absolute right-2 top-2"),
            cls="relative"
        ),
        cls="grid items-center p-3 bg-muted/80 dark:bg-muted/40 w-full",
    )


def Search():
    return Div(
        Div(
            SearchBox(),
            cls="grid"
        ),
        cls="grid",
    )
