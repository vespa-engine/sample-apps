from fasthtml.common import *
from shad4fast import *

from ui.home import Home
from ui.layout import Layout
from ui.search import Search

highlight_js_theme_link = Link(id='highlight-theme', rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(langs=['python', 'javascript', 'java', 'json', 'xml'], dark="github-dark", light="github")

app, rt = fast_app(
    htmlkw={'cls': "h-full"},
    pico=False,
    hdrs=(
        ShadHead(tw_cdn=False, theme_handle=True),
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme
    ),
)


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(f'./static/{filepath}')


@rt("/")
def get():
    return Layout(Home())


@rt("/search")
def get():
    return Layout(Search())


serve()
