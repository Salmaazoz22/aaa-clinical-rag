# -*- coding: utf-8 -*-
"""The product name, in one place.

Every visible occurrence of the product name comes from here: the browser tab
title (`ui/theme.page_config`) and the rail wordmark (`ui/components.nav_rail`).
Defined once so a rename is one edit and cannot leave a stale name behind on a
page nobody happened to open.

What this is NOT allowed to touch: module names, API routes, the corpus document
identifiers (`ESVS_2024`, `NICE_NG156`, ...), or the clinical subject matter.
"AAA" is a diagnosis, not branding, and stays wherever it names the condition.
"""
from __future__ import annotations

from typing import Final

#: The wordmark. Shown in the nav rail and in the browser tab.
PRODUCT_NAME: Final[str] = "Clinova X"

#: The line under the wordmark. Says what the product is, not what it contains.
PRODUCT_TAGLINE: Final[str] = "Clinical Evidence Intelligence"


def page_title(view_title: str) -> str:
    """The browser tab title for one view: "Ask · Clinova X"."""
    view_title = (view_title or "").strip()
    return f"{view_title} · {PRODUCT_NAME}" if view_title else PRODUCT_NAME
