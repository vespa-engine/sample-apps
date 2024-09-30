from fasthtml.common import *
from lucide_fasthtml import Lucide
from shad4fast import *

app, rt = fast_app(
    hdrs=(ShadHead(tw_cdn=False, theme_handle=True)),
    pico=False,
    htmlkw={'cls': "h-full"}

)


def Logo():
    return Div(
        Img(src='https://assets.vespa.ai/logos/vespa-logo-black.svg', alt='Vespa Logo', cls='h-full dark:hidden'),
        Img(src='https://assets.vespa.ai/logos/vespa-logo-white.svg', alt='Vespa Logo Dark Mode',
            cls='h-full hidden dark:block'),
        cls='h-[27px]'
    )


def ThemeToggle(variant="ghost", cls=None, **kwargs):
    return Button(
        Lucide("sun", cls="dark:flex hidden"),
        Lucide("moon", cls="dark:hidden"),
        variant=variant,
        size="icon",
        cls=f"theme-toggle {cls}",
        **kwargs,
    )


def Links():
    return Nav(
        A(
            Button(Lucide(icon="github"), size="icon", variant="ghost"),
            href="https://github.com/vespa-engine/vespa",
            target="_blank",
        ),
        A(
            Button(Lucide(icon="slack"), size="icon", variant="ghost"),
            href="https://slack.vespa.ai",
            target="_blank",
        ),
        Separator(orientation="vertical"),
        ThemeToggle(),
        cls='flex items-center space-x-3'
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


def SearchBox():
    return Div(
        Textarea(
            placeholder="Enter your search query...",
            cls="max-h-[377px] border-transparent ring-offset-transparent ring-0 focus-visible:ring-transparent text-base resize-y overflow-hidden appearance-none",
        ),
        Div(
            Span(
                "tests",
                cls="text-muted-foreground"
            ),
            Span(
                Button(Lucide(icon="arrow-right"), size="sm"),
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


def Layout(*c, **kwargs):
    return (
        Title('Visual Retrieval ColPali'),
        Body(
            Header(
                Logo(),
                Links(),
                cls='h-[55px] w-full flex items-center justify-between px-4'
            ),
            Main(
                *c, **kwargs,
                cls='flex-1 h-full p-4'
            ),
            cls='min-h-screen h-full flex flex-col'
        ),
    )


@rt("/")
def get():
    return Layout(Home())


serve()
