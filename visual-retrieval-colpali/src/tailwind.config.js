function filterDefault(values) {
    return Object.fromEntries(
        Object.entries(values).filter(([key]) => key !== "DEFAULT"),
    );
}

/** @type {import('tailwindcss').Config} */
export default {
    darkMode: ["selector"],
    content: [
        "../**/*.py",
        "../.venv/lib/python3.12/site-packages/shad4fast/**/*.{py,js}",
    ],
    theme: {
        container: {
            center: true,
            padding: "2rem",
            screens: {
                "2xl": "1400px",
            },
        },
        extend: {
            animation: {
                "accordion-down": "accordion-down 0.2s ease-out",
                "accordion-up": "accordion-up 0.2s ease-out",
            },
            animationDelay: ({theme}) => ({
                ...theme("transitionDelay"),
            }),
            animationDuration: ({theme}) => ({
                0: "0ms",
                ...theme("transitionDuration"),
            }),
            animationTimingFunction: ({theme}) => ({
                ...theme("transitionTimingFunction"),
            }),
            animationFillMode: {
                none: "none",
                forwards: "forwards",
                backwards: "backwards",
                both: "both",
            },
            animationDirection: {
                normal: "normal",
                reverse: "reverse",
                alternate: "alternate",
                "alternate-reverse": "alternate-reverse",
            },
            animationOpacity: ({theme}) => ({
                DEFAULT: 0,
                ...theme("opacity"),
            }),
            animationTranslate: ({theme}) => ({
                DEFAULT: "100%",
                ...theme("translate"),
            }),
            animationScale: ({theme}) => ({
                DEFAULT: 0,
                ...theme("scale"),
            }),
            animationRotate: ({theme}) => ({
                DEFAULT: "30deg",
                ...theme("rotate"),
            }),
            animationRepeat: {
                0: "0",
                1: "1",
                infinite: "infinite",
            },
            keyframes: {
                enter: {
                    from: {
                        opacity: "var(--tw-enter-opacity, 1)",
                        transform:
                            "translate3d(var(--tw-enter-translate-x, 0), var(--tw-enter-translate-y, 0), 0) scale3d(var(--tw-enter-scale, 1), var(--tw-enter-scale, 1), var(--tw-enter-scale, 1)) rotate(var(--tw-enter-rotate, 0))",
                    },
                },
                exit: {
                    to: {
                        opacity: "var(--tw-exit-opacity, 1)",
                        transform:
                            "translate3d(var(--tw-exit-translate-x, 0), var(--tw-exit-translate-y, 0), 0) scale3d(var(--tw-exit-scale, 1), var(--tw-exit-scale, 1), var(--tw-exit-scale, 1)) rotate(var(--tw-exit-rotate, 0))",
                    },
                },
            },
            colors: {
                border: "hsl(var(--border))",
                input: "hsl(var(--input))",
                ring: "hsl(var(--ring))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",
                primary: {
                    DEFAULT: "hsl(var(--primary))",
                    foreground: "hsl(var(--primary-foreground))",
                },
                secondary: {
                    DEFAULT: "hsl(var(--secondary))",
                    foreground: "hsl(var(--secondary-foreground))",
                },
                destructive: {
                    DEFAULT: "hsl(var(--destructive))",
                    foreground: "hsl(var(--destructive-foreground))",
                },
                muted: {
                    DEFAULT: "hsl(var(--muted))",
                    foreground: "hsl(var(--muted-foreground))",
                },
                accent: {
                    DEFAULT: "hsl(var(--accent))",
                    foreground: "hsl(var(--accent-foreground))",
                },
                popover: {
                    DEFAULT: "hsl(var(--popover))",
                    foreground: "hsl(var(--popover-foreground))",
                },
                card: {
                    DEFAULT: "hsl(var(--card))",
                    foreground: "hsl(var(--card-foreground))",
                },
            },
            borderRadius: {
                lg: `var(--radius)`,
                md: `calc(var(--radius) - 2px)`,
                sm: "calc(var(--radius) - 4px)",
            },
        },
    },
    plugins: [
        function ({addUtilities, matchUtilities, theme}) {
            addUtilities({
                "@keyframes enter": theme("keyframes.enter"),
                "@keyframes exit": theme("keyframes.exit"),
                ".animate-in": {
                    animationName: "enter",
                    animationDuration: theme("animationDuration.DEFAULT"),
                    "--tw-enter-opacity": "initial",
                    "--tw-enter-scale": "initial",
                    "--tw-enter-rotate": "initial",
                    "--tw-enter-translate-x": "initial",
                    "--tw-enter-translate-y": "initial",
                },
                ".animate-out": {
                    animationName: "exit",
                    animationDuration: theme("animationDuration.DEFAULT"),
                    "--tw-exit-opacity": "initial",
                    "--tw-exit-scale": "initial",
                    "--tw-exit-rotate": "initial",
                    "--tw-exit-translate-x": "initial",
                    "--tw-exit-translate-y": "initial",
                },
            });

            matchUtilities(
                {
                    "fade-in": (value) => ({"--tw-enter-opacity": value}),
                    "fade-out": (value) => ({"--tw-exit-opacity": value}),
                },
                {values: theme("animationOpacity")},
            );

            matchUtilities(
                {
                    "zoom-in": (value) => ({"--tw-enter-scale": value}),
                    "zoom-out": (value) => ({"--tw-exit-scale": value}),
                },
                {values: theme("animationScale")},
            );

            matchUtilities(
                {
                    "spin-in": (value) => ({"--tw-enter-rotate": value}),
                    "spin-out": (value) => ({"--tw-exit-rotate": value}),
                },
                {values: theme("animationRotate")},
            );

            matchUtilities(
                {
                    "slide-in-from-top": (value) => ({
                        "--tw-enter-translate-y": `-${value}`,
                    }),
                    "slide-in-from-bottom": (value) => ({
                        "--tw-enter-translate-y": value,
                    }),
                    "slide-in-from-left": (value) => ({
                        "--tw-enter-translate-x": `-${value}`,
                    }),
                    "slide-in-from-right": (value) => ({
                        "--tw-enter-translate-x": value,
                    }),
                    "slide-out-to-top": (value) => ({
                        "--tw-exit-translate-y": `-${value}`,
                    }),
                    "slide-out-to-bottom": (value) => ({
                        "--tw-exit-translate-y": value,
                    }),
                    "slide-out-to-left": (value) => ({
                        "--tw-exit-translate-x": `-${value}`,
                    }),
                    "slide-out-to-right": (value) => ({
                        "--tw-exit-translate-x": value,
                    }),
                },
                {values: theme("animationTranslate")},
            );

            matchUtilities(
                {duration: (value) => ({animationDuration: value})},
                {values: filterDefault(theme("animationDuration"))},
            );

            matchUtilities(
                {delay: (value) => ({animationDelay: value})},
                {values: theme("animationDelay")},
            );

            matchUtilities(
                {ease: (value) => ({animationTimingFunction: value})},
                {values: filterDefault(theme("animationTimingFunction"))},
            );

            addUtilities({
                ".running": {animationPlayState: "running"},
                ".paused": {animationPlayState: "paused"},
            });

            matchUtilities(
                {"fill-mode": (value) => ({animationFillMode: value})},
                {values: theme("animationFillMode")},
            );

            matchUtilities(
                {direction: (value) => ({animationDirection: value})},
                {values: theme("animationDirection")},
            );

            matchUtilities(
                {repeat: (value) => ({animationIterationCount: value})},
                {values: theme("animationRepeat")},
            );
        },
    ],
};
