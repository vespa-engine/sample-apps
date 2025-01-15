(function () {
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
    observer.observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
})();
