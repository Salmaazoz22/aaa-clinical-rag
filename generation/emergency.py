# -*- coding: utf-8 -*-
"""Pre-retrieval emergency safety gate for potential acute aortic presentations.

This gate runs BEFORE any other check — including the patient-specific gate in
`generation/safety.py` — so that a query describing a plausible aortic emergency
is always redirected to emergency services, regardless of whether it is also
patient-specific.  Priority: emergency > patient-specific > all other gates.

What this module does (and does NOT do):
  - DOES: detect queries that combine acute-presentation language (sudden/severe
    pain, collapse, haemodynamic instability) with AAA-relevant context cues,
    and flag them so the pipeline can return an urgent emergency-services redirect.
  - Does NOT: diagnose a rupture.
  - Does NOT: provide reassurance that the situation is or is not an emergency.
  - Does NOT: provide individualised treatment direction.
  - Does NOT: cite guideline text (the calling pipeline omits citations, for the
    same reason the patient-specific gate does: citing guidelines here would look
    like an answer to an individual's acute situation).

Design mirrors `generation/safety.py`:
  - Signal groups (named regex tuples) → rule over signal combinations → one
    screening function returning a frozen verdict dataclass.
  - Deterministic, no model call, no network call.
  - Conservative: avoid a single-keyword trigger.  An acute-symptom signal
    MUST co-occur with an AAA-context cue to fire the gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
#
# Two independent groups are required:
#
#   1. ACUTE_SYMPTOM — describes something happening RIGHT NOW that is severe or
#      sudden.  None of these fire on informational phrasing ("what causes back
#      pain in AAA patients?").
#
#   2. AAA_CONTEXT — ties the question to an aortic aneurysm (existing diagnosis,
#      named rupture, or typical rupture language) so we don't gate every question
#      that mentions "sudden pain" in the abstract.
#
# The gate fires when AT LEAST ONE pattern from EACH group matches.

_ACUTE_SYMPTOM: tuple[re.Pattern[str], ...] = (
    # sudden / acute onset
    re.compile(r"\bsudden\b", re.I),
    re.compile(r"\bacute\s+onset\b", re.I),
    re.compile(r"\bcoming\s+on\s+(fast|quickly|suddenly|rapidly)\b", re.I),
    # tearing / ripping / stabbing pain character (classic aortic-dissection/rupture descriptors)
    re.compile(r"\btearing\s+pain\b", re.I),
    re.compile(r"\bripping\s+pain\b", re.I),
    re.compile(r"\bstabbing\s+pain\b", re.I),
    re.compile(r"\bexcruciating\s+pain\b", re.I),
    # severe pain in abdomen / back / flank
    re.compile(r"\bsevere\b.{0,30}\b(abdominal|back|flank|loin)\s+pain\b", re.I),
    re.compile(r"\b(abdominal|back|flank|loin)\s+pain\b.{0,30}\bsevere\b", re.I),
    # collapse / syncope / loss of consciousness
    re.compile(r"\b(collapsed|collapsing|syncope|syncopal|fainted|fainting|passed\s+out)\b", re.I),
    re.compile(r"\bloss\s+of\s+consciousness\b", re.I),
    # haemodynamic instability
    re.compile(r"\bhypotension\b", re.I),
    re.compile(r"\bshock\b.{0,20}\b(blood|BP|pressure|haemodynamic|hemodynamic)\b", re.I),
    re.compile(r"\b(blood\s+pressure|BP)\b.{0,20}\b(low|dropped|dropping|crash)\b", re.I),
    # unwell / deteriorating — combined with context cue, strong indicator
    re.compile(r"\b(rapidly?\s+)?deteriorat\w+\b", re.I),
    # rupture / ruptured — this IS the acute event (not just context)
    re.compile(r"\b(rupture|ruptured|rupturing)\b", re.I),
)

_AAA_CONTEXT: tuple[re.Pattern[str], ...] = (
    # named diagnosis / existing aneurysm
    re.compile(r"\b(known|diagnosed|existing|confirmed)\b.{0,30}\b(aneurysm|aaa)\b", re.I),
    re.compile(r"\b(aneurysm|aaa)\b.{0,30}\b(known|diagnosed|existing|confirmed)\b", re.I),
    # possessive — "my aneurysm", "his AAA", "her aortic aneurysm"
    re.compile(
        r"\b(my|his|her|their|our|the)\s+(aortic\s+)?(abdominal\s+aortic\s+)?aneurysm\b", re.I
    ),
    re.compile(r"\b(my|his|her|their|our|the)\s+aaa\b", re.I),
    # rupture / ruptured — explicit clinical language
    re.compile(r"\b(rupture|ruptured|rupturing)\b", re.I),
    # typical rupture presentation: back/abdominal pain + AAA anywhere in sentence
    re.compile(r"\b(aneurysm|aaa)\b", re.I),  # broad: fires only when ACUTE_SYMPTOM also fires
    # aortic emergency language
    re.compile(r"\baortic\s+(emergency|crisis|catastrophe)\b", re.I),
    # pulsatile / pulsating abdominal mass
    re.compile(r"\bpulsatile\b.{0,30}\b(mass|lump|swelling)\b", re.I),
)


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmergencyVerdict:
    """Outcome of screening one query for potential emergency presentation."""

    is_emergency: bool
    acute_signals: list[str] = field(default_factory=list)
    context_signals: list[str] = field(default_factory=list)
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "is_emergency": self.is_emergency,
            "acute_signals": list(self.acute_signals),
            "context_signals": list(self.context_signals),
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------

def screen_emergency(query: str) -> EmergencyVerdict:
    """Decide whether this query describes a potential acute aortic emergency.

    Fires when at least one pattern from EACH of the two signal groups matches:
      - ACUTE_SYMPTOM: something severe/sudden is happening right now.
      - AAA_CONTEXT: the query is anchored to an aortic aneurysm.

    A bare mention of "sudden pain" with no AAA context does NOT fire.
    A bare mention of "known AAA" with no acute symptom does NOT fire.
    Both must co-occur.

    Returns an `EmergencyVerdict` with `is_emergency=True` only when the gate
    fires.  The detail string is intended for internal audit only — it is never
    shown to the end user as a diagnosis or clinical assessment.
    """
    if not isinstance(query, str):
        return EmergencyVerdict(is_emergency=False)

    acute_hits = [p.pattern for p in _ACUTE_SYMPTOM if p.search(query)]
    context_hits = [p.pattern for p in _AAA_CONTEXT if p.search(query)]

    if acute_hits and context_hits:
        return EmergencyVerdict(
            is_emergency=True,
            acute_signals=acute_hits,
            context_signals=context_hits,
            detail=(
                "Query combines an acute-symptom indicator with an AAA-context cue. "
                "This system cannot assess whether this is an emergency."
            ),
        )

    return EmergencyVerdict(
        is_emergency=False,
        acute_signals=acute_hits,
        context_signals=context_hits,
    )
