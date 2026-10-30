# genpark-cyber-dating

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-zero-success)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

GenPark Cyber Dating is a zero-dependency algorithmic matchmaking and partner recommendation engine implementing the Nobel-prize-winning Gale-Shapley Deferred Acceptance stable matching algorithm, Big Five personality trait vector cosine affinity modeling, and bilateral dynamic ELO ratings.

```mermaid
flowchart TD
    A[Proposer Preference Lists] --> B[Gale-Shapley Deferred Acceptance Engine]
    C[Receiver Preference Lists] --> B
    B --> D[Strictly Stable Bilateral Pairings]
    E[Big Five Personality Profiles] --> F[Cosine & Weighted Euclidean Affinity]
    F --> G[Soulmate Compatibility Score]
    H[Interaction Outcome / Match Swipes] --> I[Bilateral Dynamic ELO Updater]
    I --> J[Equilibrium Preference Convergence]
```

## Features
- **Gale-Shapley Stable Matching**: Guarantees zero blocking pairs and Pareto optimality across multi-agent pairing rosters.
- **Big Five Trait Vector Affinity**: Measures cosine similarity and weighted Euclidean distance across Ocean personality dimensions.
- **Bilateral ELO Rating System**: Computes expected preference probability and adapts agent desirability ratings dynamically.

## Quickstart

```python
from client import GenParkCyberDating

dating = GenParkCyberDating()
res = dating.solve_gale_shapley_stable_matching(
    {"A": ["X", "Y"], "B": ["Y", "X"]},
    {"X": ["B", "A"], "Y": ["A", "B"]}
)
print("Matches:", res["matched_pairs"])
```

## License
MIT
