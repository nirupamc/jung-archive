"""Curated seed vocabulary + deterministic concept normalization (M7).

This is a SEED vocabulary, not a hardcoded ontology; new entries only
add nodes/aliases. Normalization is deterministic: NFKC, casefold,
punctuation stripping, whitespace collapse, then alias mapping.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

VOCAB_VERSION = "jung-vocab-1"


@dataclass(frozen=True)
class Concept:
    canonical_name: str
    node_type: str = "CONCEPT"
    aliases: tuple = ()
    description: str = ""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    return " ".join(s.split())


VOCABULARY = [
    Concept("Shadow", "ARCHETYPE",
            ("the shadow", "shadow archetype", "darker side", "dark side"),
            "The unrecognized / inferior part of the personality."),
    Concept("Self", "ARCHETYPE", ("the self", "true self"),
            "The central archetype of order and wholeness."),
    Concept("Persona", "ARCHETYPE", ("the persona",),
            "The outward social mask of the individual."),
    Concept("Ego", "CONCEPT", ("the ego",)),
    Concept("Anima", "ARCHETYPE", ("the anima",)),
    Concept("Animus", "ARCHETYPE", ("the animus",)),
    Concept("Individuation", "CONCEPT", ("individuation process",),
            "Becoming an individual whole being."),
    Concept("Collective Unconscious", "CONCEPT",
            ("the collective unconscious",)),
    Concept("Personal Unconscious", "CONCEPT",
            ("the personal unconscious",)),
    Concept("Synchronicity", "CONCEPT", ("synchronistic",)),
    Concept("Archetype", "ARCHETYPE",
            ("archetypes", "archetypal", "primordial image")),
    Concept("Projection", "CONCEPT", ("projected", "projections")),
    Concept("Complex", "CONCEPT", ("complexes", "feeling-toned complex")),
    Concept("Mandala", "SYMBOL", ("mandalas",)),
    Concept("Dream", "SYMBOL", ("dreams", "dream symbolism")),
    Concept("Consciousness", "CONCEPT", ("conscious mind", "the conscious")),
    Concept("Unconscious", "CONCEPT", ("the unconscious", "unconscious psyche")),
    Concept("Religion", "THEME", ("religious", "creed", "creeds", "churches",
                                  "the church", "faith")),
    Concept("Mass-mindedness", "THEME",
            ("mass mindedness", "mass-mindedness", "mass man", "mass rule",
             "the mass", "masses", "mass psychology", "mass action",
             "organized mass", "crowd")),
    Concept("Self-knowledge", "THEME",
            ("self knowledge", "self-examination", "know thyself")),
    Concept("State", "THEME", ("the state", "dictator state", "nation state",
                               "state slavery", "raison d'etat")),
    Concept("Individual", "THEME", ("the individual", "individuality",
                                    "individual man")),
    Concept("Nihilism", "THEME", ("nihilistic despair",)),
    Concept("God", "SYMBOL", ("god-image", "image of god", "divine")),
]


@dataclass
class Vocabulary:
    concepts: List[Concept] = field(default_factory=list)

    @property
    def version(self) -> str:
        return VOCAB_VERSION

    def __post_init__(self):
        if not self.concepts:
            self.concepts = list(VOCABULARY)
        # deterministic lookup tables
        self.by_normalized: dict = {}
        self.alias_to_canonical: dict = {}
        for c in self.concepts:
            self.by_normalized[_norm(c.canonical_name)] = c
            for alias in (c.canonical_name, *c.aliases):
                self.alias_to_canonical[_norm(alias)] = c.canonical_name

    def canonical(self, surface: str):
        """Map any surface form to its canonical concept name (or None)."""
        return self.alias_to_canonical.get(_norm(surface))

    def find_mentions(self, text: str) -> List[dict]:
        """Find all vocabulary mentions in text.

        Deterministic: longest-alias-first scanning on normalized text.
        Returns [{canonical, surface, char_start, char_end}] using the
        normalized-text offsets (stable for ordering/dedup).
        """
        norm = _norm(text)
        mentions = []
        claimed = [False] * len(norm)
        # longest first so "collective unconscious" beats "unconscious"
        for alias in sorted(self.alias_to_canonical,
                            key=len, reverse=True):
            start = 0
            while True:
                idx = norm.find(alias, start)
                if idx == -1:
                    break
                end = idx + len(alias)
                boundary_ok = (
                    (idx == 0 or not (norm[idx - 1].isalnum()))
                    and (end == len(norm) or not norm[end].isalnum())
                )
                if boundary_ok and not any(claimed[idx:end]):
                    for i in range(idx, end):
                        claimed[i] = True
                    mentions.append({
                        "canonical": self.alias_to_canonical[alias],
                        "surface": alias,
                        "char_start": idx,
                        "char_end": end,
                    })
                start = idx + 1
        mentions.sort(key=lambda m: m["char_start"])
        return mentions


def normalize_name(surface: str) -> str:
    return _norm(surface)


def node_id_for(canonical_name: str) -> str:
    """Deterministic node id: 'concept:<normalized-name>'."""
    return f"concept:{_norm(canonical_name)}"
