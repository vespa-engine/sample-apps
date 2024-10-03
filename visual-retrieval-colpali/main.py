from fasthtml.common import *
from shad4fast import *
import json

from ui.home import Home
from ui.layout import Layout
from ui.search import Search
from backend.vespa_app import get_vespa_app
from backend.colpali import load_model, get_result_dummy
from vespa.application import Vespa

highlight_theme_link = Link(id="highlight-theme", rel="stylesheet", href="")

theme_script = Script("""
    (function() {
        function getPreferredTheme() {
            if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                return 'dark';
            }
            return 'light';
        }

        function syncHighlightTheme() {
            const link = document.getElementById('highlight-theme');
            const preferredTheme = getPreferredTheme();
            link.href = preferredTheme === 'dark' ? 
                'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/styles/github-dark.min.css' :
                'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.4.0/styles/github.min.css';
        }

        // Apply the correct theme immediately
        syncHighlightTheme();

        // Observe changes in the 'dark' class on the <html> element
        const observer = new MutationObserver(syncHighlightTheme);
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    })();
""")

highlight_js = HighlightJS(
    langs=["python", "javascript", "java", "json", "xml"],
    dark="github-dark",
    light="github",
)

app, rt = fast_app(
    hdrs=(
        ShadHead(tw_cdn=False, theme_handle=True),
        highlight_js,
        highlight_theme_link,
        theme_script,
    ),
    pico=False,
    htmlkw={"cls": "h-full"},
)
vespa_app: Vespa = get_vespa_app()

# Initialize variables to None
model = None
processor = None


# Define a function to get the model and processor
def get_model_and_processor():
    global model, processor
    if model is None or processor is None:
        model, processor = load_model()
    return model, processor


@rt("/")
def get():
    return Layout(Home())


@rt("/search")
def get():
    return Layout(Search())


@rt("/app")
def get():
    return Layout(Div(P(f"Connected to Vespa at {app.url}")))


@rt("/run_query")
def get(query: str, nn: bool = False):
    # dummy-function to avoid running the query every time
    result = get_result_dummy(query, nn)
    # If we want to run real, uncomment the following lines
    # model, processor = get_model_and_processor()
    # result = asyncio.run(
    # get_result_from_query(vespa_app, processor=processor, model=model, query=query, nn=nn)
    # )
    return Layout(
        Div(
            H1("Result"),
            Pre(Code(json.dumps(result))),
        )
    )


serve()
