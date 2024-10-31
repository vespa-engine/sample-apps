from fasthtml.components import Body, Div, Header, Img, Nav, Title
from fasthtml.xtend import A, Script
from lucide_fasthtml import Lucide
from shad4fast import Button, Separator

layout_script = Script(
    """
    document.addEventListener("DOMContentLoaded", function () {
          const main = document.querySelector('main');
          const aside = document.querySelector('aside');
          const body = document.body;
        
          if (main && aside && main.nextElementSibling === aside) {
            // If we have both main and aside, adjust the layout for larger screens
            body.classList.remove('grid-cols-1'); // Remove single-column layout
            body.classList.add('md:grid-cols-[minmax(0,_45fr)_minmax(0,_15fr)]'); // Two-column layout on larger screens
          } else if (main) {
            // If only main, keep it full width
            body.classList.add('grid-cols-1');
          }
    });
    """
)

overlay_scrollbars_manager = Script(
    """
    (function () {
        const { OverlayScrollbars } = OverlayScrollbarsGlobal;

        function getPreferredTheme() {
            return localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)
                ? 'dark'
                : 'light';
        }

        function applyOverlayScrollbars(element, scrollbarTheme) {
            // Destroy existing OverlayScrollbars instance if it exists
            const instance = OverlayScrollbars(element);
            if (instance) {
                instance.destroy();
            }

            // Reinitialize OverlayScrollbars with the correct theme and settings
            OverlayScrollbars(element, {
                overflow: {
                    x: 'hidden',
                    y: 'scroll'
                },
                scrollbars: {
                    theme: scrollbarTheme,
                    visibility: 'auto',
                    autoHide: 'leave',
                    autoHideDelay: 800
                }
            });
        }

        // Function to get the current scrollbar theme (light or dark)
        function getScrollbarTheme() {
            const isDarkMode = getPreferredTheme() === 'dark';
            return isDarkMode ? 'os-theme-light' : 'os-theme-dark';  // Light theme in dark mode, dark theme in light mode
        }

        // Expose the common functions globally for reuse
        window.OverlayScrollbarsManager = {
            applyOverlayScrollbars: applyOverlayScrollbars,
            getScrollbarTheme: getScrollbarTheme
        };
    })();
    """
)

static_elements_scrollbars = Script(
    """
    (function () {
        const { applyOverlayScrollbars, getScrollbarTheme } = OverlayScrollbarsManager;

        function applyScrollbarsToStaticElements() {
            const mainElement = document.querySelector('main');
            const chatMessagesElement = document.querySelector('#chat-messages');

            const scrollbarTheme = getScrollbarTheme();

            if (mainElement) {
                applyOverlayScrollbars(mainElement, scrollbarTheme);
            }

            if (chatMessagesElement) {
                applyOverlayScrollbars(chatMessagesElement, scrollbarTheme);
            }
        }

        // Apply the scrollbars on page load
        applyScrollbarsToStaticElements();

        // Observe changes in the 'dark' class on the <html> element to adjust the theme dynamically
        const observer = new MutationObserver(applyScrollbarsToStaticElements);
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    })();
    """
)


def Logo():
    return Div(
        Img(
            src="https://assets.vespa.ai/logos/vespa-logo-black.svg",
            alt="Vespa Logo",
            cls="h-full dark:hidden",
        ),
        Img(
            src="https://assets.vespa.ai/logos/vespa-logo-white.svg",
            alt="Vespa Logo Dark Mode",
            cls="h-full hidden dark:block",
        ),
        cls="h-[27px]",
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
            Button("About this demo?", variant="link"),
            href="/about-this-demo",
        ),
        Separator(orientation="vertical"),
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
        cls="flex items-center space-x-2",
    )


def Layout(*c, is_home=False, **kwargs):
    return (
        Title("Visual Retrieval ColPali"),
        Body(
            Header(
                A(Logo(), href="/"),
                Links(),
                cls="min-h-[55px] h-[55px] w-full flex items-center justify-between px-4",
            ),
            *c,
            **kwargs,
            data_is_home=str(is_home).lower(),
            cls="grid grid-rows-[minmax(0,55px)_minmax(0,1fr)] min-h-0",
        ),
        layout_script,
        overlay_scrollbars_manager,
        static_elements_scrollbars,
    )
