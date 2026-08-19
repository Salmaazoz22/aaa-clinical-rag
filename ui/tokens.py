# -*- coding: utf-8 -*-
"""Design tokens — the only place a colour or a measure is defined.

No other module in `ui/` may contain a raw hex value. `assets/theme.css` declares
the same values as CSS custom properties; this module exists so Python-side
component code can reference them by name, and so a test can assert that the
stylesheet, the Streamlit native theme in `.streamlit/config.toml`, and this file
never drift apart.

The palette is six named values. Everything else is a neutral derived from them
or a status colour kept deliberately outside the brand.

    --ink       near-black with a cyan cast; ink on film
    --slate     the instrument chrome: nav rail, footer band
    --linen     the reading canvas; cool clinical linen
    --surface   cards, composer, evidence cards
    --contrast  THE brand accent, named for contrast media — the agent that
                makes vessels visible. Interaction, focus, active nav, the
                caliper threshold marker. Never a warning.
    --aorta     oxidized vessel red. Thresholds, abstention, critical only.
"""
from __future__ import annotations

from typing import Final

# --- the six ---------------------------------------------------------------

INK: Final = "#0C1418"
SLATE: Final = "#16242A"
LINEN: Final = "#F1F4F3"
SURFACE: Final = "#FFFFFF"
CONTRAST: Final = "#C0803A"
AORTA: Final = "#9E3B3E"

#: `--contrast` lifted for legibility on `--slate`. The base amber sits at
#: roughly 3.4:1 on slate; this variant clears 4.5:1.
CONTRAST_LIFTED: Final = "#E0A855"

# --- derived neutrals ------------------------------------------------------

LINE: Final = "#D8DEDC"
LINE_DARK: Final = "rgba(255,255,255,0.09)"
MUTED: Final = "#5C6B70"
MUTED_DARK: Final = "#8FA3A9"

#: One step between --slate and --surface, for the sidebar's own inset surfaces.
SLATE_RAISED: Final = "#1E3038"
SLATE_BORDER: Final = "#2A3C43"

# --- status, kept separate from the brand ----------------------------------

VERIFIED: Final = "#3B7355"
CAUTION: Final = AORTA          # caution is the aorta family, never amber
NEUTRAL: Final = "#4A6572"

# --- type ------------------------------------------------------------------

SANS: Final = '"Instrument Sans", "Inter Tight", Inter, system-ui, sans-serif'
SERIF: Final = '"Source Serif 4", "Iowan Old Style", Georgia, serif'
MONO: Final = '"IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace'

#: rem, against a 16px root.
SCALE: Final = {
    "display": 3.0,
    "h1": 2.0,
    "h2": 1.5,
    "h3": 1.25,
    "body": 1.0625,
    "base": 1.0,
    "small": 0.875,
    "eyebrow": 0.75,
}

# --- space, radius, elevation ----------------------------------------------

#: 4px base.
SPACE: Final = (4, 8, 12, 16, 24, 32, 48, 64, 96)

RADIUS_CONTROL: Final = "4px"
RADIUS_DATA: Final = "2px"
RADIUS_PILL: Final = "999px"

SHADOW: Final = "0 1px 2px rgba(12,20,24,.06), 0 10px 28px -14px rgba(12,20,24,.20)"

# --- layout ----------------------------------------------------------------

RAIL_W: Final = 232          # nav rail, px
CANVAS_MAX: Final = 1440     # canvas max width, px
HEADER_H: Final = 56         # canvas header, px
MEASURE: Final = "68ch"      # max line length for clinical prose

#: Breakpoints the layout is verified at.
BREAKPOINTS: Final = (1440, 1280, 1024, 834, 390)


def as_css_variables() -> str:
    """The token block for `:root`, generated so it cannot drift from this file."""
    pairs = (
        ("ink", INK), ("slate", SLATE), ("linen", LINEN), ("surface", SURFACE),
        ("contrast", CONTRAST), ("contrast-lifted", CONTRAST_LIFTED), ("aorta", AORTA),
        ("line", LINE), ("line-dark", LINE_DARK), ("muted", MUTED), ("muted-dark", MUTED_DARK),
        ("slate-raised", SLATE_RAISED), ("slate-border", SLATE_BORDER),
        ("verified", VERIFIED), ("caution", CAUTION), ("neutral", NEUTRAL),
        ("sans", SANS), ("serif", SERIF), ("mono", MONO),
        ("r-control", RADIUS_CONTROL), ("r-data", RADIUS_DATA), ("r-pill", RADIUS_PILL),
        ("shadow", SHADOW),
        ("rail-w", f"{RAIL_W}px"), ("canvas-max", f"{CANVAS_MAX}px"),
        ("header-h", f"{HEADER_H}px"), ("measure", MEASURE),
    )
    return "\n".join(f"  --{name}: {value};" for name, value in pairs)


#: Mirror of the values written into `.streamlit/config.toml`. A test asserts
#: the file still carries exactly these, so the native theme and the stylesheet
#: cannot drift.
NATIVE_THEME_MIRROR: Final = {
    "primaryColor": CONTRAST,
    "backgroundColor": LINEN,
    "secondaryBackgroundColor": SURFACE,
    "textColor": INK,
    "borderColor": LINE,
    "greenColor": VERIFIED,
    "redColor": AORTA,
    "blueColor": NEUTRAL,
}

NATIVE_SIDEBAR_MIRROR: Final = {
    "backgroundColor": SLATE,
    "secondaryBackgroundColor": SLATE_RAISED,
    "primaryColor": CONTRAST_LIFTED,
    "borderColor": SLATE_BORDER,
}
