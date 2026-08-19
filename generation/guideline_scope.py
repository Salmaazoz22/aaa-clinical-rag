# -*- coding: utf-8 -*-
"""Pre-generation gate for guideline editions the corpus does not contain.

A question can name a guideline that does not exist here — "what does the 2026
ESVS guideline recommend?" — and still retrieve strongly, because every chunk in
the index is an ESVS/NICE/SVS/USPSTF passage about exactly that topic. The
similarity floor cannot catch it: the retrieval is *good*, it is simply the
wrong edition. Left to the model, the refusal depends on the model noticing the
year, being reachable, and being willing to say no — three things that are not
guaranteed, and the third of which is the one a clinical system must not
outsource.

So the check is made here, deterministically, from the corpus's own metadata:

  * the editions available are read from `data/processed/document_metadata.json`,
    the same file `/v1/corpus` serves. Nothing is hardcoded; adding a document to
    the corpus changes what this gate allows, with no edit here;
  * a year only counts as an *edition request* when a guideline or organisation
    cue sits beside it. "The 2013 Bown study" names a year and is not a request
    for a 2013 guideline, so it passes;
  * when an organisation is named, only that organisation's editions are
    considered. "2026 ESVS" is refused because ESVS 2024 is the only ESVS
    edition indexed — not because 2026 is in the future.

Like `generation/safety.py` and `generation/emergency.py`: signal groups, a rule
over them, one screening function, a frozen verdict. No model call, no network.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: The corpus manifest. The same file `api/main.py` serves at `/v1/corpus`.
CORPUS_METADATA_PATH = ROOT / "data" / "processed" / "document_metadata.json"

#: A four-digit year that could plausibly be a guideline edition.
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

#: NICE-style guidance identifiers ("NG156", "ng 156", "NG-156").
_GUIDANCE_ID = re.compile(r"\b([a-z]{2,4})\s*-?\s*(\d{2,4})\b", re.I)

#: Words that turn a nearby year into a request for a specific *edition*.
_EDITION_CUE = re.compile(
    r"\b(?:guideline|guidelines|guidance|recommendation|recommendations|"
    r"statement|edition|version|update|updated|revision|revised|consensus|"
    r"practice\s+guideline\w*|clinical\s+practice)\b",
    re.I,
)

#: How far either side of a year an edition cue may sit and still bind to it.
#: Kept short, and further bounded by `_CLAUSE_BREAK`: "the 2026 ESVS guideline"
#: binds, "a trial published in 2021 ...; what does the guidance say" does not.
_CUE_WINDOW = 40

#: Punctuation that ends the clause a year belongs to. A cue on the far side of
#: one of these is talking about something else.
_CLAUSE_BREAK = re.compile(r"[.;:?!]")

#: Identifier prefixes that are guidance-document identifiers rather than random
#: letter-digit pairs. Kept narrow on purpose: a bare `\w+\d+` match would fire
#: on "AAA 55" or "CTA 3".
_GUIDANCE_ID_PREFIXES = frozenset({"ng", "cg", "qs", "ta", "dg", "ipg", "mtg"})


@dataclass(frozen=True)
class CorpusEditions:
    """Which guideline editions the indexed corpus actually contains."""

    #: organisation alias (lower-case) -> the years that organisation has here.
    years_by_organisation: dict[str, frozenset[int]] = field(default_factory=dict)
    #: every year present, from any organisation.
    all_years: frozenset[int] = frozenset()
    #: normalised guidance identifiers present, e.g. {"ng156"}.
    identifiers: frozenset[str] = frozenset()
    #: human-readable list, e.g. ["ESVS 2024", "NICE NG156 (2020)", ...].
    labels: tuple[str, ...] = ()

    def years_for(self, aliases: Iterable[str]) -> frozenset[int]:
        """The union of editions available for the named organisations.

        A union rather than an intersection: a question naming two
        organisations ("compare the 2024 ESVS and 2019 USPSTF advice") is asking
        about both, and every year it names is available from one of them.
        """
        found: set[int] = set()
        for alias in aliases:
            found |= set(self.years_by_organisation.get(alias, frozenset()))
        return frozenset(found)

    @property
    def known_aliases(self) -> frozenset[str]:
        return frozenset(self.years_by_organisation)


@dataclass(frozen=True)
class GuidelineScopeVerdict:
    """Outcome of screening one question for an unavailable guideline edition."""

    blocked: bool
    requested: tuple[str, ...] = ()
    available: tuple[str, ...] = ()
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "requested": list(self.requested),
            "available": list(self.available),
            "detail": self.detail,
        }


def _aliases_for(document_id: str, organisation: str) -> set[str]:
    """Every way a question might name the organisation behind one document.

    Two sources, both already in the manifest: the acronym that prefixes the
    `document_id` (`ESVS_2024` -> "esvs"), and any acronym the organisation
    string carries in parentheses ("... Surgery (ESVS)" -> "esvs"). The full
    organisation name is added as well so a spelled-out question still matches.
    """
    aliases: set[str] = set()
    head = str(document_id).split("_", 1)[0].strip().lower()
    if head:
        aliases.add(head)
    org = str(organisation or "").strip()
    if org:
        aliases.add(org.lower())
        for acronym in re.findall(r"\(([A-Za-z]{2,10})\)", org):
            aliases.add(acronym.lower())
        # "National Institute for Health and Care Excellence (NICE)" also has a
        # short common name before the parenthesis; keep it too.
        without_paren = re.sub(r"\s*\([^)]*\)\s*", " ", org).strip().lower()
        if without_paren:
            aliases.add(without_paren)
    return {a for a in aliases if a}


def _identifiers_in(*values: Any) -> set[str]:
    """Normalised guidance identifiers ("ng156") found in the given strings."""
    found: set[str] = set()
    for value in values:
        for prefix, digits in _GUIDANCE_ID.findall(str(value or "")):
            if prefix.lower() in _GUIDANCE_ID_PREFIXES:
                found.add(f"{prefix.lower()}{digits}")
    return found


@lru_cache(maxsize=4)
def load_corpus_editions(path: str | None = None) -> CorpusEditions:
    """Read the available editions from the corpus manifest.

    Cached: the manifest is a build-time artifact that cannot change while the
    process is up. A missing or unreadable manifest yields an EMPTY
    `CorpusEditions`, and `screen_guideline_edition` refuses nothing when the
    editions are unknown — a gate that cannot see the corpus must not guess.
    """
    manifest = Path(path) if path else CORPUS_METADATA_PATH
    try:
        documents = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return CorpusEditions()
    if not isinstance(documents, list):
        return CorpusEditions()

    by_alias: dict[str, set[int]] = {}
    all_years: set[int] = set()
    identifiers: set[str] = set()
    labels: list[str] = []

    for document in documents:
        if not isinstance(document, dict):
            continue
        document_id = str(document.get("document_id") or "")
        organisation = document.get("source_organization") or ""

        years: set[int] = set()
        published = document.get("publication_year")
        if isinstance(published, int):
            years.add(published)
        years |= {int(y) for y in _YEAR.findall(document_id)}
        all_years |= years

        for alias in _aliases_for(document_id, organisation):
            by_alias.setdefault(alias, set()).update(years)

        identifiers |= _identifiers_in(document_id, document.get("source_url"))
        if document_id:
            label = document_id.replace("_", " ")
            if years and not _YEAR.search(label):
                label += f" ({sorted(years)[0]})"
            labels.append(label)

    return CorpusEditions(
        years_by_organisation={a: frozenset(y) for a, y in by_alias.items()},
        all_years=frozenset(all_years),
        identifiers=frozenset(identifiers),
        labels=tuple(labels),
    )


def _year_is_edition_request(query: str, match: re.Match[str], aliases: Sequence[str]) -> bool:
    """Does this year read as a request for a guideline edition?

    True when an edition cue ("guideline", "recommendations", "update") or an
    organisation alias sits within `_CUE_WINDOW` characters of the year. A year
    standing alone, or attached to a study or a statistic, is not an edition
    request and must not be gated (task requirement: do not reject a question
    merely because it contains a year).
    """
    start = max(0, match.start() - _CUE_WINDOW)
    end = min(len(query), match.end() + _CUE_WINDOW)
    # Clip the window at the nearest clause break on either side of the year.
    breaks_before = [m.end() for m in _CLAUSE_BREAK.finditer(query, start, match.start())]
    if breaks_before:
        start = breaks_before[-1]
    break_after = _CLAUSE_BREAK.search(query, match.end(), end)
    if break_after:
        end = break_after.start()
    window = query[start:end]
    if _EDITION_CUE.search(window):
        return True
    lowered = window.lower()
    return any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases)


def screen_guideline_edition(
    query: str, corpus: CorpusEditions | None = None
) -> GuidelineScopeVerdict:
    """Decide whether the question asks for a guideline edition we do not have.

    Blocks when, and only when:

    E1  the question names a year that reads as a guideline edition, and that
        year is not among the editions the corpus contains (scoped to the
        organisation named, when one is named);
    E2  the question names a guidance identifier (NICE "NG###") that is not the
        one indexed.

    Everything else is allowed through, including questions with no year, years
    attached to studies rather than guidelines, and every edition the corpus
    does contain.
    """
    if not isinstance(query, str) or not query.strip():
        return GuidelineScopeVerdict(blocked=False)

    editions = corpus if corpus is not None else load_corpus_editions()
    if not editions.all_years and not editions.identifiers:
        # The manifest could not be read. Refusing on an unknown corpus would be
        # guessing, so this gate stands down.
        return GuidelineScopeVerdict(blocked=False)

    lowered = query.lower()
    named_aliases = sorted(
        alias
        for alias in editions.known_aliases
        if re.search(rf"\b{re.escape(alias)}\b", lowered)
    )
    allowed_years = editions.years_for(named_aliases) if named_aliases else editions.all_years

    # E1 -- an edition year we do not carry.
    unavailable_years = []
    for match in _YEAR.finditer(query):
        if not _year_is_edition_request(query, match, named_aliases):
            continue
        year = int(match.group(0))
        if year not in allowed_years:
            unavailable_years.append(year)

    if unavailable_years:
        organisation = named_aliases[0].upper() if named_aliases else None
        requested = [
            f"{organisation} {year}" if organisation else str(year)
            for year in sorted(set(unavailable_years))
        ]
        return GuidelineScopeVerdict(
            blocked=True,
            requested=tuple(requested),
            available=editions.labels,
            detail=(
                f"the question asks for {', '.join(requested)}, which is not among the "
                f"guideline editions in the indexed corpus"
            ),
        )

    # E2 -- a guidance identifier we do not carry.
    requested_ids = _identifiers_in(query)
    unknown_ids = sorted(i for i in requested_ids if i not in editions.identifiers)
    if unknown_ids and editions.identifiers:
        return GuidelineScopeVerdict(
            blocked=True,
            requested=tuple(i.upper() for i in unknown_ids),
            available=editions.labels,
            detail=(
                f"the question asks for {', '.join(i.upper() for i in unknown_ids)}, which "
                f"is not among the guidance documents in the indexed corpus"
            ),
        )

    return GuidelineScopeVerdict(blocked=False, available=editions.labels)
