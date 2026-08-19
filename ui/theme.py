# -*- coding: utf-8 -*-
"""The visual language: one stylesheet, one palette, one set of primitives.

Clinical/research aesthetic, not chat UI. IBM Plex Serif for headings, Plex Sans
for body, Plex Mono for anything a reader might need to copy or verify — chunk
IDs, digests, similarity scores, model revisions.

Colour carries meaning and nothing else:

    GREEN   verified / grounded / supported by retrieved evidence
    AMBER   caution — a warning-level validator finding, a weak score, a caveat
    RED     refused / blocked / failed — never used decoratively
"""
from __future__ import annotations

import streamlit as st

# --- palette ---------------------------------------------------------------

BG = "#F6F7F9"
SURFACE = "#FFFFFF"
INK = "#13171E"
INK_MUTED = "#5A6472"
INK_FAINT = "#8A93A0"
ACCENT = "#2B4C9B"
ACCENT_SOFT = "#EEF2FB"
BORDER = "#E1E5EC"
BORDER_STRONG = "#CBD2DE"

GREEN = "#1B7F4B"
GREEN_SOFT = "#E9F5EE"
AMBER = "#9A6511"
AMBER_SOFT = "#FDF3E2"
RED = "#B3261E"
RED_SOFT = "#FCEDEC"

SERIF = "'IBM Plex Serif', Georgia, 'Times New Roman', serif"
SANS = "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"

FONT_LINK = (
    "https://fonts.googleapis.com/css2"
    "?family=IBM+Plex+Sans:wght@400;500;600;700"
    "&family=IBM+Plex+Serif:wght@400;500;600;700"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&display=swap"
)

PAGE_ICON = "🩺"


def _fonts() -> str:
    """The Google Fonts <link>s, as their own payload.

    Kept separate from the stylesheet on purpose. `<link>` is a CommonMark
    *type 6* HTML block, and a type 6 block ENDS AT THE FIRST BLANK LINE. When
    the fonts and the stylesheet shared one payload, the block opened on
    `<link>` and closed at the first blank line inside the CSS -- so every rule
    after that point was parsed as Markdown prose and rendered as visible text
    on the page. This payload contains no blank line, so it stays one HTML
    block from first character to last.
    """
    return "\n".join((
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        f'<link href="{FONT_LINK}" rel="stylesheet">',
    ))


def _css() -> str:
    """The stylesheet, as a payload that BEGINS with `<style>`.

    `<style>` is a CommonMark *type 1* HTML block, and a type 1 block ends only
    at its matching closing tag -- blank lines inside it are safe. That is the
    whole reason this string starts at column zero with `<style>` and nothing
    else: it is what keeps the CSS inside the element instead of on the page.

    Do not prepend anything to this return value.
    """
    return f"""<style>
  :root {{
    --bg: {BG};  --surface: {SURFACE};  --ink: {INK};  --ink-muted: {INK_MUTED};
    --ink-faint: {INK_FAINT}; --accent: {ACCENT}; --accent-soft: {ACCENT_SOFT};
    --border: {BORDER}; --border-strong: {BORDER_STRONG};
    --green: {GREEN}; --green-soft: {GREEN_SOFT};
    --amber: {AMBER}; --amber-soft: {AMBER_SOFT};
    --red: {RED};   --red-soft: {RED_SOFT};
  }}

  .stApp {{ background: var(--bg); }}
  html, body, [class*="st-"], .stMarkdown, p, li, div, label, input, textarea, button {{
    font-family: {SANS};
    color: var(--ink);
  }}
  .block-container {{ padding-top: 2.4rem; padding-bottom: 5rem; max-width: 1180px; }}

  h1, h2, h3, h4 {{ font-family: {SERIF}; color: var(--ink); letter-spacing: -0.01em; }}
  h1 {{ font-weight: 600; }}
  h2 {{ font-weight: 600; font-size: 1.5rem; margin-top: 2.2rem; }}
  h3 {{ font-weight: 600; font-size: 1.15rem; }}
  code, kbd, pre, .mono {{ font-family: {MONO}; font-size: 0.83rem; }}

  /* --- sidebar ---------------------------------------------------------- */
  section[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

  /* --- masthead --------------------------------------------------------- */
  .masthead {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1.75rem 2rem 1.5rem;
    margin-bottom: 1.25rem;
  }}
  .masthead .eyebrow {{
    font-family: {MONO}; font-size: 0.7rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); font-weight: 600;
  }}
  .masthead h1 {{ font-size: 2.1rem; margin: 0.35rem 0 0.2rem; line-height: 1.15; }}
  .masthead .tagline {{
    font-family: {SERIF}; font-size: 1.08rem; color: var(--ink-muted);
    margin: 0 0 0.85rem;
  }}
  .masthead .lede {{
    font-size: 0.92rem; color: var(--ink-muted); max-width: 62ch;
    line-height: 1.6; margin: 0;
  }}

  /* --- badges ----------------------------------------------------------- */
  .badge-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 1.1rem; }}
  .badge {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: {MONO}; font-size: 0.72rem; font-weight: 500;
    padding: 0.3rem 0.6rem; border-radius: 4px;
    border: 1px solid var(--border-strong); background: var(--bg); color: var(--ink-muted);
    white-space: nowrap;
  }}
  .badge b {{ color: var(--ink); font-weight: 600; }}
  .badge.ok    {{ border-color: #BFE0CD; background: var(--green-soft); color: var(--green); }}
  .badge.ok b  {{ color: var(--green); }}
  .badge.warn  {{ border-color: #EFD9AE; background: var(--amber-soft); color: var(--amber); }}
  .badge.warn b{{ color: var(--amber); }}
  .badge.bad   {{ border-color: #EFC5C2; background: var(--red-soft); color: var(--red); }}
  .badge.bad b {{ color: var(--red); }}
  .badge.accent{{ border-color: #C6D2EC; background: var(--accent-soft); color: var(--accent); }}
  .badge.accent b {{ color: var(--accent); }}

  /* --- cards ------------------------------------------------------------ */
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.25rem 1.4rem; margin-bottom: 1rem;
  }}
  .card.tight {{ padding: 0.9rem 1.1rem; }}
  .card-label {{
    font-family: {MONO}; font-size: 0.68rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--ink-faint); font-weight: 600;
    margin-bottom: 0.55rem;
  }}
  .rule {{ height:1px; background: var(--border); border:0; margin: 1.6rem 0 1.1rem; }}

  /* --- answer ----------------------------------------------------------- */
  .answer {{
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 6px;
    padding: 1.5rem 1.7rem; margin-bottom: 1rem;
  }}
  .answer.refused {{ border-left-color: var(--amber); background: #FFFDF9; }}
  .answer .body {{ font-size: 1.02rem; line-height: 1.68; color: var(--ink); }}
  .answer .body p {{ margin: 0 0 0.7rem; }}

  .disclaimer {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 5px; padding: 0.85rem 1rem;
    font-size: 0.79rem; line-height: 1.55; color: var(--ink-muted);
  }}
  .disclaimer b {{ color: var(--ink); }}

  /* --- stat tiles ------------------------------------------------------- */
  .stat {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.85rem 1rem; height: 100%;
  }}
  .stat .k {{
    font-family: {MONO}; font-size: 0.66rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ink-faint); font-weight: 600;
  }}
  .stat .v {{
    font-family: {MONO}; font-size: 1.45rem; font-weight: 600;
    color: var(--ink); line-height: 1.25; margin-top: 0.25rem;
  }}
  .stat .v.green {{ color: var(--green); }}
  .stat .v.amber {{ color: var(--amber); }}
  .stat .v.red   {{ color: var(--red); }}
  .stat .v.accent{{ color: var(--accent); }}
  .stat .s {{ font-size: 0.74rem; color: var(--ink-muted); margin-top: 0.2rem; }}

  /* --- evidence --------------------------------------------------------- */
  .ev-head {{ display:flex; align-items:baseline; gap:0.6rem; flex-wrap:wrap; }}
  .ev-rank {{
    font-family: {MONO}; font-size: 0.72rem; font-weight: 600;
    color: var(--accent); background: var(--accent-soft);
    border: 1px solid #C6D2EC; border-radius: 3px; padding: 0.1rem 0.4rem;
  }}
  .ev-doc {{ font-family: {SERIF}; font-weight: 600; font-size: 0.95rem; }}
  .ev-meta {{ font-family: {MONO}; font-size: 0.72rem; color: var(--ink-faint); }}
  .ev-text {{
    font-family: {SERIF}; font-size: 0.9rem; line-height: 1.7;
    color: #23282F; background: var(--bg);
    border: 1px solid var(--border); border-left: 3px solid var(--border-strong);
    border-radius: 4px; padding: 0.9rem 1.1rem; margin-top: 0.6rem;
    white-space: pre-wrap;
  }}
  .ev-text.used {{ border-left-color: var(--green); }}
  .ev-text.dropped {{ border-left-color: var(--border-strong); color: var(--ink-muted); }}

  /* --- score bars ------------------------------------------------------- */
  .bars {{ margin-top: 0.4rem; }}
  .bar-row {{
    display: grid; grid-template-columns: 62px 1fr 74px;
    align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;
  }}
  .bar-lab {{ font-family: {MONO}; font-size: 0.72rem; color: var(--ink-faint); }}
  .bar-track {{
    position: relative; height: 20px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 3px; overflow: hidden;
  }}
  .bar-fill {{ height: 100%; }}
  .bar-fill.used {{ background: {GREEN}; }}
  .bar-fill.dropped {{ background: {BORDER_STRONG}; }}
  .bar-val {{ font-family: {MONO}; font-size: 0.76rem; font-weight: 600; text-align: right; }}
  .bar-val.used {{ color: var(--green); }}
  .bar-val.dropped {{ color: var(--ink-faint); }}
  .thresh {{ position:absolute; top:0; bottom:0; width:2px; background: {RED}; opacity:0.75; }}
  .thresh-note {{
    font-family: {MONO}; font-size: 0.7rem; color: var(--red);
    margin-top: 0.35rem;
  }}

  /* --- pipeline trace --------------------------------------------------- */
  .trace {{ display:flex; flex-direction:column; gap:0; }}
  .trace-step {{
    display:grid; grid-template-columns: 26px 1fr; gap:0.75rem;
    padding: 0.55rem 0; border-bottom: 1px dashed var(--border);
  }}
  .trace-step:last-child {{ border-bottom: 0; }}
  .trace-dot {{
    width: 18px; height: 18px; border-radius: 50%; margin-top: 2px;
    display:flex; align-items:center; justify-content:center;
    font-size: 0.66rem; font-weight: 700; color: #fff;
  }}
  .trace-dot.pass {{ background: var(--green); }}
  .trace-dot.stop {{ background: var(--amber); }}
  .trace-dot.skip {{ background: var(--border-strong); color: var(--ink-faint); }}
  .trace-dot.fail {{ background: var(--red); }}
  .trace-name {{ font-weight: 600; font-size: 0.88rem; }}
  .trace-detail {{ font-family: {MONO}; font-size: 0.75rem; color: var(--ink-muted); margin-top: 0.12rem; }}

  /* --- citation trace --------------------------------------------------- */
  .trace-chain {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.1rem 1.25rem; margin-bottom: 0.85rem;
  }}
  .chain-claim {{
    font-family: {SERIF}; font-size: 0.95rem; line-height: 1.6;
    padding-bottom: 0.7rem; border-bottom: 1px solid var(--border); margin-bottom: 0.7rem;
  }}
  .chain-row {{
    display: grid; grid-template-columns: 96px 1fr; gap: 0.7rem;
    padding: 0.22rem 0; align-items: baseline;
  }}
  .chain-k {{
    font-family: {MONO}; font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-faint); font-weight: 600;
  }}
  .chain-v {{ font-family: {MONO}; font-size: 0.8rem; color: var(--ink); word-break: break-all; }}
  .chain-v.serif {{ font-family: {SERIF}; font-size: 0.92rem; word-break: normal; }}
  .checks {{ display:flex; flex-wrap:wrap; gap:0.35rem; margin-top:0.75rem; }}
  .check {{
    font-family: {MONO}; font-size: 0.7rem; font-weight: 500;
    padding: 0.2rem 0.45rem; border-radius: 3px; white-space: nowrap;
  }}
  .check.y {{ background: var(--green-soft); color: var(--green); border: 1px solid #BFE0CD; }}
  .check.n {{ background: var(--red-soft);   color: var(--red);   border: 1px solid #EFC5C2; }}
  .check.w {{ background: var(--amber-soft); color: var(--amber); border: 1px solid #EFD9AE; }}
  .check.o {{ background: var(--bg); color: var(--ink-faint); border: 1px solid var(--border); }}

  /* --- notices ---------------------------------------------------------- */
  .notice {{
    border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    border: 1px solid var(--border); background: var(--surface);
    font-size: 0.89rem; line-height: 1.6;
  }}
  .notice .t {{ font-weight: 600; margin-bottom: 0.3rem; display:block; }}
  .notice.info {{ border-left: 4px solid var(--accent);  background: var(--accent-soft); }}
  .notice.good {{ border-left: 4px solid var(--green);   background: var(--green-soft); }}
  .notice.warn {{ border-left: 4px solid var(--amber);   background: var(--amber-soft); }}
  .notice.bad  {{ border-left: 4px solid var(--red);     background: var(--red-soft); }}
  .notice code {{ background: rgba(19,23,30,0.06); padding: 0.08rem 0.3rem; border-radius: 3px; }}

  /* --- tables ----------------------------------------------------------- */
  .tbl {{ width:100%; border-collapse: collapse; font-size: 0.83rem; }}
  .tbl th {{
    font-family: {MONO}; font-size: 0.68rem; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--ink-faint); font-weight: 600;
    text-align: right; padding: 0.5rem 0.6rem; border-bottom: 2px solid var(--border-strong);
  }}
  .tbl th:first-child, .tbl td:first-child {{ text-align: left; }}
  .tbl td {{
    font-family: {MONO}; padding: 0.45rem 0.6rem; text-align: right;
    border-bottom: 1px solid var(--border); color: var(--ink);
  }}
  .tbl tr.shipped td {{ background: var(--accent-soft); font-weight: 600; }}
  .tbl tr.shipped td:first-child {{ color: var(--accent); }}
  .tbl-wrap {{ overflow-x: auto; }}

  /* --- misc ------------------------------------------------------------- */
  .kv {{ display:grid; grid-template-columns: 190px 1fr; gap: 0.35rem 0.9rem; font-size: 0.83rem; }}
  .kv .k {{ font-family: {MONO}; font-size: 0.72rem; color: var(--ink-faint);
            text-transform: uppercase; letter-spacing: 0.06em; padding-top: 0.12rem; }}
  .kv .v {{ font-family: {MONO}; font-size: 0.8rem; word-break: break-all; }}
  .kv .v.wrap {{ font-family: {SANS}; font-size: 0.85rem; word-break: normal; line-height:1.55; }}

  .footnote {{ font-size: 0.78rem; color: var(--ink-faint); line-height: 1.6; }}

  div[data-testid="stExpander"] details {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: 0.55rem;
  }}
  div[data-testid="stExpander"] summary {{ font-size: 0.88rem; }}

  .stButton > button {{
    border-radius: 5px; font-weight: 500; font-size: 0.88rem;
    border: 1px solid var(--border-strong);
  }}
  .stButton > button[kind="primary"] {{
    background: var(--accent); border-color: var(--accent);
  }}
  .stTextArea textarea, .stTextInput input {{
    font-family: {SANS}; font-size: 0.95rem;
    border-radius: 5px; border-color: var(--border-strong); background: var(--surface);
  }}
  [data-testid="stMetricValue"] {{ font-family: {MONO}; }}
</style>"""


def apply(page_title: str) -> None:
    """Configure the page and inject the stylesheet. Call once, first, per page."""
    st.set_page_config(
        page_title=f"{page_title} — AAA Clinical RAG",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Two separate calls, deliberately. Concatenating them would re-create the
    # type 6 / blank line bug described in _fonts() and _css().
    st.markdown(_fonts(), unsafe_allow_html=True)
    st.markdown(_css(), unsafe_allow_html=True)
