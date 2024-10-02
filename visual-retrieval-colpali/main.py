from fasthtml.common import *
from shad4fast import *

from ui.home import Home
from ui.layout import Layout
from ui.search import Search

highlight_theme_link = Link(
    id='highlight-theme',
    rel="stylesheet",
    href=""
)

theme_script = Script('''
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
''')

highlight_js = HighlightJS(langs=['python', 'javascript', 'java', 'json', 'xml'], dark="github-dark", light="github")

app, rt = fast_app(
    hdrs=(ShadHead(tw_cdn=False, theme_handle=True), highlight_js, highlight_theme_link, theme_script),
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
