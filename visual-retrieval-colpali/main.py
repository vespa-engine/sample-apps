from fasthtml.common import *
from shad4fast import *

app, rt = fast_app(pico=False, hdrs=(ShadHead(tw_cdn=True),))


@rt("/")
def get():
    return Titled("Hello World!", Alert(title="Shad4Fast", description="You're all set up!"))


serve()
