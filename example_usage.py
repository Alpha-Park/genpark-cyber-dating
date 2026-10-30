# -*- coding: utf-8 -*-
import json
from client import GenParkCyberDating

dating = GenParkCyberDating()

# 1. Gale-Shapley Stable Marriage Matching
proposers = {
    "Agent_Alpha": ["Agent_Echo", "Agent_Foxtrot", "Agent_Golf"],
    "Agent_Beta": ["Agent_Foxtrot", "Agent_Echo", "Agent_Golf"],
    "Agent_Gamma": ["Agent_Echo", "Agent_Golf", "Agent_Foxtrot"]
}
receivers = {
    "Agent_Echo": ["Agent_Beta", "Agent_Alpha", "Agent_Gamma"],
    "Agent_Foxtrot": ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"],
    "Agent_Golf": ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"]
}
matching = dating.solve_gale_shapley_stable_matching(proposers, receivers)
print(f"Gale-Shapley Stable Matching: Stable={matching['is_stable_matching']} ({matching['iterations']} iterations)")
for prop, rec in matching["matched_pairs"].items():
    print(f"  {prop} <---> {rec}")

# 2. Big Five Personality Trait Semantic Affinity
profile_a = {"openness": 0.85, "conscientiousness": 0.70, "extraversion": 0.60, "agreeableness": 0.80, "neuroticism": 0.20}
profile_b = {"openness": 0.80, "conscientiousness": 0.75, "extraversion": 0.65, "agreeableness": 0.85, "neuroticism": 0.25}
affinity = dating.calculate_semantic_affinity_vector(profile_a, profile_b)
print()
print(f"Personality Affinity: Score={affinity['composite_affinity_score']}/100 ({affinity['match_grade']})")
print(f"  Cosine Sim: {affinity['cosine_similarity']} | Euclidean Dist: {affinity['weighted_euclidean_distance']}")

# 3. Bilateral ELO Dynamic Update
elo = dating.update_bilateral_elo_ratings(rating_a=1500.0, rating_b=1580.0, outcome_a=1.0)
print()
print(f"Bilateral ELO Update: Agent A ${elo['initial_ratings']['a']} -> ${elo['updated_ratings']['a']} (+{elo['elo_deltas']['delta_a']})")
print(f"  Agent B ${elo['initial_ratings']['b']} -> ${elo['updated_ratings']['b']} ({elo['elo_deltas']['delta_b']})")
