from fasthtml.components import Div, Img, Nav, Title, Body, Header, Main
from fasthtml.xtend import A
from lucide_fasthtml import Lucide
from shad4fast import Button, Separator


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


def Layout(*c, **kwargs):
    return (
        Title('Visual Retrieval ColPali'),
        Body(
            Header(
                A(Logo(), href="/"),
                Links(),
                cls='min-h-[55px] h-[55px] w-full flex items-center justify-between px-4'
            ),
            Main(
                *c, **kwargs,
                cls='flex-1 h-full'
            ),
            cls='h-full flex flex-col'
        ),
    )
