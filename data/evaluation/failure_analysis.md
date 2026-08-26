# Failure Analysis

## bm25_only_win (4)

- **q001_mass_mindedness** — mass-mindedness
  - ground truth: `381d2da4b68e-c00035, 381d2da4b68e-c00045, 381d2da4b68e-c00051`
  - {'dense_first_rank': None, 'bm25_first_rank': 5}
- **q007_protection_from_mass** — What protects an individual from being absorbed into the psychology of the mass?
  - ground truth: `381d2da4b68e-c00036, 381d2da4b68e-c00103, 381d2da4b68e-c00111, 381d2da4b68e-c00164`
  - {'dense_first_rank': None, 'bm25_first_rank': 7}
- **q013_two_edged_weapon** — What does Jung call a two-edged weapon and why?
  - ground truth: `381d2da4b68e-c00056`
  - {'dense_first_rank': None, 'bm25_first_rank': 4}
- **q028_religious_advantage** — What advantage does the religious person have for answering crucial questions?
  - ground truth: `381d2da4b68e-c00168`
  - {'dense_first_rank': None, 'bm25_first_rank': 1}

## reranker_hurts_rank (2)

- **q006_lose_independence** — Why does the individual lose independence in a mass society?
  - ground truth: `381d2da4b68e-c00029, 381d2da4b68e-c00030, 381d2da4b68e-c00032, 381d2da4b68e-c00100`
  - {'hybrid_rank': 1, 'rerank_rank': 2}
- **q022_word_became_god** — What does Jung say has happened to 'the Word' in Western thought?
  - ground truth: `381d2da4b68e-c00140, 381d2da4b68e-c00141`
  - {'hybrid_rank': 2, 'rerank_rank': 3}

## reranker_improves_rank (10)

- **q001_mass_mindedness** — mass-mindedness
  - ground truth: `381d2da4b68e-c00035, 381d2da4b68e-c00045, 381d2da4b68e-c00051`
  - {'hybrid_rank': 6, 'rerank_rank': 4}
- **q002_self_knowledge** — self-knowledge
  - ground truth: `381d2da4b68e-c00011, 381d2da4b68e-c00012, 381d2da4b68e-c00166, 381d2da4b68e-c00201`
  - {'hybrid_rank': 7, 'rerank_rank': 1}
- **q003_metamorphosis_gods** — What does Jung mean by a 'metamorphosis of the gods'?
  - ground truth: `381d2da4b68e-c00208`
  - {'hybrid_rank': 2, 'rerank_rank': 1}
- **q005_iron_curtain_split** — What split is symbolized by the 'Iron Curtain'?
  - ground truth: `381d2da4b68e-c00005, 381d2da4b68e-c00121`
  - {'hybrid_rank': 2, 'rerank_rank': 1}
- **q013_two_edged_weapon** — What does Jung call a two-edged weapon and why?
  - ground truth: `381d2da4b68e-c00056`
  - {'hybrid_rank': 10, 'rerank_rank': 1}
- **q015_projection_of_evil** — Why does Jung think people project evil onto others?
  - ground truth: `381d2da4b68e-c00178, 381d2da4b68e-c00184`
  - {'hybrid_rank': 7, 'rerank_rank': 3}
- **q017_state_replaces_god** — How does Jung describe the State taking the place of God?
  - ground truth: `381d2da4b68e-c00045, 381d2da4b68e-c00046`
  - {'hybrid_rank': 2, 'rerank_rank': 1}
- **q023_instinct_neglect_medical** — What happens when instinct is violated or neglected?
  - ground truth: `381d2da4b68e-c00155`
  - {'hybrid_rank': 4, 'rerank_rank': 1}
- **q024_zeitgeist_modern_art** — What role does Jung give to modern art and the unconscious Zeitgeist?
  - ground truth: `381d2da4b68e-c00205`
  - {'hybrid_rank': 2, 'rerank_rank': 1}
- **q027_freedom_foundation** — On what foundation do freedom and autonomy of the individual rest?
  - ground truth: `381d2da4b68e-c00041, 381d2da4b68e-c00042`
  - {'hybrid_rank': 2, 'rerank_rank': 1}
