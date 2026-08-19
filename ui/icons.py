# -*- coding: utf-8 -*-
"""Inline SVG icons. No icon font, no emoji, no `:material/` shortcodes.

This module is the permanent fix for the `_arrow_right` defect. Streamlit draws
Material icons as `<span data-testid="stIconMaterial">arrow_right</span>` and
relies on a font *ligature* to turn that text into a glyph — so any stylesheet
that touches `font-family` on Streamlit internals makes the raw icon name
visible. Nothing here depends on a font: every icon is geometry.

All icons share a 24×24 viewBox, use `currentColor`, and inherit their size from
the caller, so an icon always matches the colour and scale of the text beside it.
"""
from __future__ import annotations

from typing import Final

_PATHS: Final[dict[str, str]] = {
    # --- navigation ------------------------------------------------------
    "ask": '<path d="M4 5h16M4 11h16M4 17h9" stroke="currentColor" stroke-width="1.6" '
           'stroke-linecap="round" fill="none"/>'
           '<circle cx="18.5" cy="17" r="2.6" stroke="currentColor" stroke-width="1.6" fill="none"/>'
           '<path d="M20.6 19.1 22.5 21" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "evaluation": '<path d="M4 20V9M9.3 20V4M14.7 20v-7M20 20v-11" stroke="currentColor" '
                  'stroke-width="1.8" stroke-linecap="round" fill="none"/>',
    "safety": '<path d="M12 3.2 19.5 6v6c0 4.3-3 7.6-7.5 8.9C7.5 19.6 4.5 16.3 4.5 12V6z" '
              'stroke="currentColor" stroke-width="1.6" fill="none" stroke-linejoin="round"/>'
              '<path d="m8.9 12.1 2.2 2.2 4-4.4" stroke="currentColor" stroke-width="1.6" '
              'fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "architecture": '<rect x="3.5" y="3.5" width="7" height="7" rx="1" stroke="currentColor" '
                    'stroke-width="1.5" fill="none"/>'
                    '<rect x="13.5" y="3.5" width="7" height="7" rx="1" stroke="currentColor" '
                    'stroke-width="1.5" fill="none"/>'
                    '<rect x="3.5" y="13.5" width="7" height="7" rx="1" stroke="currentColor" '
                    'stroke-width="1.5" fill="none"/>'
                    '<rect x="13.5" y="13.5" width="7" height="7" rx="1" stroke="currentColor" '
                    'stroke-width="1.5" fill="none"/>'
                    '<path d="M10.5 7h3M7 10.5v3M17 10.5v3M10.5 17h3" stroke="currentColor" '
                    'stroke-width="1.5"/>',
    "sources": '<path d="M5 4.5h9.5L19 9v10.5H5z" stroke="currentColor" stroke-width="1.6" '
               'fill="none" stroke-linejoin="round"/>'
               '<path d="M14 4.5V9h5" stroke="currentColor" stroke-width="1.6" fill="none" '
               'stroke-linejoin="round"/>'
               '<path d="M8 12.5h8M8 16h5.5" stroke="currentColor" stroke-width="1.5" '
               'stroke-linecap="round"/>',
    "technical": '<circle cx="12" cy="12" r="3.1" stroke="currentColor" stroke-width="1.6" fill="none"/>'
                 '<path d="M12 3.2v3M12 17.8v3M3.2 12h3M17.8 12h3M5.8 5.8l2.1 2.1M16.1 16.1l2.1 2.1'
                 'M18.2 5.8l-2.1 2.1M7.9 16.1l-2.1 2.1" stroke="currentColor" stroke-width="1.6" '
                 'stroke-linecap="round"/>',

    # --- affordances -----------------------------------------------------
    "arrow_right": '<path d="M5 12h13m-5.5-5.5L18 12l-5.5 5.5" stroke="currentColor" '
                   'stroke-width="1.7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "chevron_down": '<path d="m6.5 9.5 5.5 5.5 5.5-5.5" stroke="currentColor" stroke-width="1.7" '
                    'fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "edit": '<path d="M4.5 19.5h4l10-10a2.1 2.1 0 0 0-3-3l-10 10z" stroke="currentColor" '
            'stroke-width="1.5" fill="none" stroke-linejoin="round"/>'
            '<path d="m13.5 6.5 4 4" stroke="currentColor" stroke-width="1.5"/>',
    "refresh": '<path d="M19.5 12a7.5 7.5 0 1 1-2.4-5.5" stroke="currentColor" stroke-width="1.7" '
               'fill="none" stroke-linecap="round"/>'
               '<path d="M19.7 4.6v4.2h-4.2" stroke="currentColor" stroke-width="1.7" fill="none" '
               'stroke-linecap="round" stroke-linejoin="round"/>',
    "close": '<path d="m6.5 6.5 11 11m0-11-11 11" stroke="currentColor" stroke-width="1.7" '
             'stroke-linecap="round"/>',
    "expand": '<path d="M4.5 9.5v-5h5M19.5 14.5v5h-5" stroke="currentColor" stroke-width="1.6" '
              'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
              '<path d="M4.5 4.5 10 10M19.5 19.5 14 14" stroke="currentColor" stroke-width="1.6"/>',

    # --- status ----------------------------------------------------------
    "check": '<path d="m5 12.5 4.5 4.5L19 7" stroke="currentColor" stroke-width="2" fill="none" '
             'stroke-linecap="round" stroke-linejoin="round"/>',
    "alert": '<path d="M12 4.5 21 20H3z" stroke="currentColor" stroke-width="1.6" fill="none" '
             'stroke-linejoin="round"/>'
             '<path d="M12 10v4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'
             '<circle cx="12" cy="17.2" r="1.05" fill="currentColor"/>',
    "info": '<circle cx="12" cy="12" r="8.4" stroke="currentColor" stroke-width="1.6" fill="none"/>'
            '<path d="M12 11v5.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>'
            '<circle cx="12" cy="7.9" r="1.05" fill="currentColor"/>',
    "offline": '<circle cx="12" cy="12" r="8.4" stroke="currentColor" stroke-width="1.6" fill="none"/>'
               '<path d="m8.4 8.4 7.2 7.2" stroke="currentColor" stroke-width="1.7" '
               'stroke-linecap="round"/>',

    # --- domain ----------------------------------------------------------
    #: The caliper: a measured value against a threshold. The app's own motif.
    "caliper": '<path d="M3 15h18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>'
               '<path d="M7 15v3M12 15v3M17 15v3" stroke="currentColor" stroke-width="1.3" '
               'stroke-linecap="round" opacity=".55"/>'
               '<rect x="13" y="6.5" width="5" height="5" rx="1" fill="currentColor"/>'
               '<path d="M9.5 4.5v13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "document": '<path d="M6 3.5h8L18.5 8v12.5H6z" stroke="currentColor" stroke-width="1.5" '
                'fill="none" stroke-linejoin="round"/>'
                '<path d="M13.8 3.5V8h4.7" stroke="currentColor" stroke-width="1.5" fill="none" '
                'stroke-linejoin="round"/>',
    "quote": '<path d="M9.4 6.5C6.9 7.8 5.5 9.9 5.5 12.6c0 2.7 1.6 4.4 3.7 4.4 1.8 0 3.1-1.3 3.1-3.1'
             '0-1.7-1.2-2.9-2.8-2.9-.3 0-.6 0-.8.1.3-1.4 1.3-2.6 2.8-3.4z" fill="currentColor"/>'
             '<path d="M18.1 6.5c-2.5 1.3-3.9 3.4-3.9 6.1 0 2.7 1.6 4.4 3.7 4.4 1.8 0 3.1-1.3 3.1-3.1'
             '0-1.7-1.2-2.9-2.8-2.9-.3 0-.6 0-.8.1.3-1.4 1.3-2.6 2.8-3.4z" fill="currentColor"/>',
}

#: Icons whose visual weight already reads as "filled"; used by the nav rail to
#: pick the right optical size.
NAMES: Final[tuple[str, ...]] = tuple(_PATHS)


def icon(name: str, size: int = 18, *, cls: str = "", title: str | None = None) -> str:
    """One inline SVG string.

    `currentColor` throughout, so the icon takes the colour of the text it sits
    with and needs no per-state variant. Decorative by default
    (`aria-hidden="true"`); pass `title` to make it announced.
    """
    path = _PATHS.get(name)
    if path is None:
        raise KeyError(f"unknown icon {name!r}; available: {', '.join(sorted(_PATHS))}")

    if title:
        a11y = f'role="img" aria-label="{title}"'
    else:
        a11y = 'aria-hidden="true" focusable="false"'

    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<svg{class_attr} width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" xmlns="http://www.w3.org/2000/svg" {a11y}>{path}</svg>'
    )
