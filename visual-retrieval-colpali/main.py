from fasthtml.common import *
from shad4fast import *

from ui.home import Home
from ui.layout import Layout
from ui.search import Search

app, rt = fast_app(
    hdrs=(ShadHead(tw_cdn=False, theme_handle=True)),
    pico=False,
    htmlkw={'cls': "h-full"}

)


@rt("/")
def get():
    return Layout(Home())


@rt("/search")
def get():
    return Layout(Search())


serve()
