/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        app:   "#09090d",
        panel: "#0e0e13",
        card:  "#13131a",
        rim:   "rgba(255,255,255,0.07)",
        accent: "#38bdf8",       // sky-400
        "accent-dim": "#0ea5e9", // sky-500
        phi:   "#7dd3fc",        // for φ labels
        field: "#a78bfa",        // for A/B/J labels
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
