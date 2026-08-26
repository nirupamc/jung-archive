"""Create the M6 benchmark dataset (manually grounded ground truth).

Ground truth was established by reading The Undiscovered Self PDF and
the chunk artifacts; chunk IDs were verified against full chunk text
(NEVER derived from the retriever under test).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jung_archive.evaluation.dataset import make_dataset_meta

DOC = "381d2da4b68e"
P = DOC + "-c%05d"


def item(qid, question, chunks, pages, tags, notes, answer=None):
    return {
        "id": qid,
        "question": question,
        "relevant_chunk_ids": [P % c if isinstance(c, int) else c for c in chunks],
        "relevant_page_numbers": pages,
        "relevant_document_ids": [DOC],
        "reference_answer": answer,
        "notes": notes,
        "tags": tags,
    }


ITEMS = [
    # ---- exact terminology -------------------------------------------------
    item("q001_mass_mindedness",
         "mass-mindedness",
         [35, 45, 51], [22, 23, 26, 28], ["exact", "terminology"],
         "Defining section opening (c00035) and core passages on the "
         "stultification/moral irresponsibility of mass-mindedness."),
    item("q002_self_knowledge",
         "self-knowledge",
         [11, 12, 166, 201], [13, 72, 84], ["exact", "terminology"],
         "Section-defining passages: people measure self-knowledge by the "
         "average person (c00011/12); rigorous self-examination (c00166); "
         "meaning of self-knowledge (c00201)."),
    item("q003_metamorphosis_gods",
         "What does Jung mean by a 'metamorphosis of the gods'?",
         [208], [87], ["exact", "conceptual"],
         "c00208 contains the only explicit 'metamorphosis of the gods' "
         "passage: change of fundamental principles and symbols driven by "
         "the unconscious man within us.",
         "A transformation of fundamental principles and symbols, "
         "expression of the changing unconscious man within us."),
    item("q004_yucca_moth",
         "yucca moth",
         [127, 128], [57, 58], ["exact", "tricky"],
         "Illustrative symbiosis example; both chunks carry the example."),
    item("q005_iron_curtain_split",
         "What split is symbolized by the 'Iron Curtain'?",
         [5, 121], [11, 55], ["exact", "cross-page"],
         "c00005 raises the symbolic split (p11); c00121 states the "
         "barbed-wire line runs through the psyche of modern man (p55)."),

    # ---- semantic paraphrases ----------------------------------------------
    item("q006_lose_independence",
         "Why does the individual lose independence in a mass society?",
         [29, 30, 32, 100], [19, 20, 48], ["semantic"],
         "Individual becomes an abstract number / statistical nullity "
         "(c00029, c00032); de-individualized society succumbs to the "
         "dictator - a million zeros add up to nothing (c00100); crowd "
         "size makes the individual negligible (c00030)."),
    item("q007_protection_from_mass",
         "What protects an individual from being absorbed into the "
         "psychology of the mass?",
         [36, 103, 111, 164], [23, 49, 52, 71], ["semantic", "conceptual"],
         "Religious standpoint enables judgment (c00036); individual "
         "relation to God as shield (c00103); resistance through "
         "individuality as organized as the mass (c00111); certainty "
         "keeping one from dissolving in the crowd (c00164)."),
    item("q008_fear_of_unconscious",
         "How does Jung describe the fear of the unconscious psyche?",
         [88, 98], [44, 47], ["semantic"],
         "Fear impedes self-knowledge and is the gravest obstacle to "
         "psychology (c00088); natural cowardice of most men (c00098)."),
    item("q009_statistics_vs_understanding",
         "Why is statistical science a problem for understanding the "
         "individual?",
         [12, 17, 20, 36], [13, 15, 16, 23], ["semantic", "conceptual"],
         "Statistical theories abolish exceptions (c00012); average-unit "
         "anthropology removes paramount individual features (c00017); "
         "individual judged a repeating unit (c00020); statistical-only "
         "reality makes judgment impossible (c00036)."),
    item("q010_band_together",
         "What happens when people band together into groups and "
         "organizations?",
         [100], [48], ["semantic"],
         "Banding together extinguishes individual personality and makes "
         "society succumb to a dictator; a million zeros do not add up "
         "to one."),
    item("q011_propaganda_eastern_europe",
         "What did Jung observe about propaganda in Eastern Europe?",
         [56, 57], [29, 30], ["section"],
         "Communal propaganda's superficial effect shown by recent events "
         "in Eastern Europe; footnote added January 1957."),
    item("q012_churches_mass_action",
         "According to Jung, how do the Churches use mass action?",
         [102], [49], ["section"],
         "Churches avail themselves of mass action to cast out the devil "
         "with Beelzebub despite caring for the individual soul."),
    item("q013_two_edged_weapon",
         "What does Jung call a two-edged weapon and why?",
         [56], [29], ["semantic"],
         "'[An] indispensable aid in the organization of masses and is "
         "therefore a two-edged weapon' (c00056; value of community "
         "depends on the spiritual and moral stature of individuals)."),
    item("q014_recognition_of_shadow",
         "What does recognition of the shadow lead to?",
         [196], [82], ["conceptual"],
         "Leads to the modesty needed to acknowledge imperfection, the "
         "ground of real human relationships.",
         "Modesty to acknowledge imperfection; basis of human relationships."),
    item("q015_projection_of_evil",
         "Why does Jung think people project evil onto others?",
         [178, 184], [76, 78], ["conceptual"],
         "We prefer to localize evil in individual criminals while "
         "washing our hands of it (c00184); the blacker-painted human "
         "shadow (c00178)."),
    item("q016_nuclear_responsibility",
         "What does Jung say about nuclear weapons and moral "
         "responsibility?",
         [189, 190], [80], ["conceptual"],
         "Reason alone does not suffice against hellish experiments; fear "
         "of the evil in one's own bosom checks reason; deciding factor "
         "lies with the individual man."),
    item("q017_state_replaces_god",
         "How does Jung describe the State taking the place of God?",
         [45, 46], [26], ["conceptual", "confusable"],
         "The dictator State swallows religious forces; State has taken "
         "the place of God; policy exalted to creed, leader becomes a "
         "demigod beyond good and evil."),
    item("q018_laboratory_psychology",
         "What is Jung's criticism of laboratory psychology?",
         [88, 89], [44, 45], ["semantic"],
         "It proceeds abstractly, removing itself from its object, so its "
         "findings are remarkably unenlightening for practical purposes."),
    item("q019_psyche_cosmic_principle",
         "In what sense does Jung call the psyche a cosmic principle?",
         [83], [42], ["conceptual", "tricky"],
         "Consciousness is a precondition of being, giving the psyche the "
         "dignity of a cosmic principle coequal with physical existence."),
    item("q020_immortality_crowd",
         "What argument about immortality does Jung attribute to the "
         "crowd?",
         [29, 30], [19], ["confusable"],
         "'They have the most convincing reason for not believing in "
         "immortality: all those people want to be immortal!' - bigger "
         "crowd, more negligible the individual."),

    # ---- section-specific ---------------------------------------------------
    item("q021_western_europe_backbone",
         "In the chapter on the West, why does Jung think Western Europe "
         "may be immune to mass seduction?",
         [73], [37], ["section"],
         "Western Europe forms the real political backbone, immune because "
         "of the outspoken counterposition of the Christian churches."),
    item("q022_word_became_god",
         "What does Jung say has happened to 'the Word' in Western "
         "thought?",
         [140, 141], [63], ["section"],
         "The word has literally become our god; universal validity severs "
         "its original link with the psyche."),

    # ---- cross-page boundary --------------------------------------------------
    item("q023_instinct_neglect_medical",
         "What happens when instinct is violated or neglected?",
         [155], [68], ["cross-page", "semantic"],
         "Painful physiological and psychological consequences requiring "
         "medical help; unconscious exists as counterbalance to "
         "consciousness."),
    item("q024_zeitgeist_modern_art",
         "What role does Jung give to modern art and the unconscious "
         "Zeitgeist?",
         [205], [86], ["section", "conceptual"],
         "Modern art performs psychological education by breaking down "
         "previous aesthetic views; the Zeitgeist compensates the "
         "conscious attitude and anticipates change."),

    # ---- confusable nearby passages -------------------------------------------
    item("q025_christ_mass_meeting",
         "Does Jung say Christ ever called his disciples at a mass "
         "meeting?",
         [103, 104], [49, 50], ["confusable"],
         "Rhetorical question pair spanning the page boundary: c00103 "
         "raises it, c00104 continues with the feeding of the five "
         "thousand."),
    item("q026_chief_factor_mass",
         "Besides huge agglomerations of people, what chief factor does "
         "Jung name for psychological mass formation?",
         [27, 28, 29], [18, 19], ["cross-page", "tricky"],
         "The passage spans the page boundary: c00028 introduces the chief "
         "factor, continuing over c00029 (loss of foundations/dignity)."),
    item("q027_freedom_foundation",
         "On what foundation do freedom and autonomy of the individual "
         "rest?",
         [41, 42], [24, 25],
         ["conceptual", "cross-page"],
         "Not ethical principles or creeds but relation to an "
         "extramundane authority acting as counterpoise to the world "
         "(spans p24-25)."),

    # ---- further coverage ------------------------------------------------------
    item("q028_religious_advantage",
         "What advantage does the religious person have for answering "
         "crucial questions?",
         [168], [73], ["section"],
         "Enjoys a great advantage: immediate experience supplies facts "
         "where others must rely on proofs."),
    item("q029_history_passes_over",
         "What does Jung predict history will do to those who resist the "
         "inevitable development?",
         [173, 174], [74, 75], ["semantic"],
         "History will pass over those who feel vocationally bound to "
         "resist; still necessary to cling to what is essential and good "
         "in tradition."),
    item("q030_kairos",
         "What Greek concept does Jung use for 'the right time'?",
         [207], [86], ["exact", "tricky"],
         "Kairos - we are living in what the Greeks called the Kairos, "
         "the right time for a metamorphosis of the gods."),
]


def main():
    artifact = json.load(
        open(f"data/chunks/{DOC}.json", encoding="utf-8"))
    sha = artifact["document"]["source_sha256"]
    meta = make_dataset_meta({DOC: sha})
    meta.dataset_version = "undiscovered-self-benchmark-1"
    dataset = {
        "meta": meta.model_dump(),
        "items": ITEMS,
    }
    out = Path("data/evaluation/dataset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {len(ITEMS)} items -> {out}")
    print(f"dataset version: {meta.dataset_version}")
    print(f"chunking config: {meta.chunking_config_version}")
    print(f"doc sha256: {sha[:16]}...")


if __name__ == "__main__":
    main()
