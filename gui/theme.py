"""Colour palette and fonts for the GUI, kept apart from the layout code."""

# A calm slate background with a teal accent. Decision colours follow the
# usual traffic-light convention.
BG = "#0f172a"          # window background (slate-900)
SURFACE = "#1e293b"     # cards (slate-800)
SURFACE_2 = "#273449"   # nested panels / preview wells
BORDER = "#334155"

ACCENT = "#2dd4bf"      # teal-400
ACCENT_HOVER = "#14b8a6"

TEXT = "#e2e8f0"        # slate-200
MUTED = "#94a3b8"       # slate-400

GOOD = "#22c55e"        # match
BAD = "#ef4444"         # different people
WARN = "#f59e0b"        # not recognised / low confidence

DECISION_COLOR = {
    "match": GOOD,
    "different": BAD,
    "unrecognized": WARN,
    "idle": MUTED,
}

DECISION_LABEL = {
    "match": "SAME PERSON",
    "different": "DIFFERENT PEOPLE",
    "unrecognized": "NOT RECOGNISED",
}

# Font families fall back gracefully if the first isn't installed.
FONT_FAMILY = "Segoe UI"
MONO_FAMILY = "Consolas"
