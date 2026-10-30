# -*- coding: utf-8 -*-
"""
GenPark Cyber Dating Client
Zero-dependency Gale-Shapley Deferred Acceptance stable matching algorithm,
Big Five personality trait semantic affinity vector solver, and bilateral ELO rating engine.
"""

import math
from typing import Dict, List, Any, Optional

class GenParkCyberDating:
    """
    Implements Gale-Shapley Nobel-prize stable matching, Big Five personality cosine affinity,
    and bilateral dynamic ELO ratings for autonomous agent pairing.
    """

    BIG_FIVE_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    def solve_gale_shapley_stable_matching(
        self,
        proposers_preferences: Dict[str, List[str]],
        receivers_preferences: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Computes stable marriage matching using the Gale-Shapley Deferred Acceptance algorithm.
        Guarantees Pareto-optimal stability with ZERO blocking pairs.
        """
        # Free proposers queue
        free_proposers = list(proposers_preferences.keys())
        # Proposer proposal index tracking
        proposals_made = {p: 0 for p in free_proposers}
        # Receiver current match: receiver -> proposer
        current_engagements = {}

        # Precompute receiver rankings: receiver -> {proposer: rank} (lower rank = more preferred)
        receiver_rankings = {}
        for r, prefs in receivers_preferences.items():
            receiver_rankings[r] = {p: idx for idx, p in enumerate(prefs)}

        iterations = 0
        max_iterations = len(free_proposers) * len(receivers_preferences) + 10

        while free_proposers and iterations < max_iterations:
            iterations += 1
            proposer = free_proposers.pop(0)
            pref_list = proposers_preferences.get(proposer, [])

            if proposals_made[proposer] < len(pref_list):
                receiver = pref_list[proposals_made[proposer]]
                proposals_made[proposer] += 1

                if receiver not in current_engagements:
                    # Receiver was free
                    current_engagements[receiver] = proposer
                else:
                    # Receiver decides between current partner and new proposer
                    current_partner = current_engagements[receiver]
                    rankings = receiver_rankings.get(receiver, {})
                    rank_current = rankings.get(current_partner, 999999)
                    rank_new = rankings.get(proposer, 999999)

                    if rank_new < rank_current:
                        # Receiver prefers new proposer
                        current_engagements[receiver] = proposer
                        free_proposers.append(current_partner)
                    else:
                        # Receiver rejects new proposer
                        free_proposers.append(proposer)

        # Invert to proposer -> receiver
        matched_pairs = {p: r for r, p in current_engagements.items()}

        # Verify stability: check for blocking pairs (m, w)
        blocking_pairs = []
        for p, r in matched_pairs.items():
            p_pref = proposers_preferences.get(p, [])
            p_r_rank = p_pref.index(r) if r in p_pref else len(p_pref)
            # Higher preferred candidates than r
            preferred_receivers = p_pref[:p_r_rank]
            for pref_r in preferred_receivers:
                curr_match_of_pref_r = current_engagements.get(pref_r)
                if curr_match_of_pref_r:
                    r_rankings = receiver_rankings.get(pref_r, {})
                    if r_rankings.get(p, 999999) < r_rankings.get(curr_match_of_pref_r, 999999):
                        blocking_pairs.append({"proposer": p, "receiver": pref_r})

        return {
            "matched_pairs": matched_pairs,
            "total_matches": len(matched_pairs),
            "iterations": iterations,
            "is_stable_matching": len(blocking_pairs) == 0,
            "blocking_pairs_count": len(blocking_pairs)
        }

    def calculate_semantic_affinity_vector(
        self,
        traits_profile_a: Dict[str, float],
        traits_profile_b: Dict[str, float],
        importance_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates cosine similarity and weighted Euclidean distance across Big Five personality traits.
        Traits scale: [0.0, 1.0]
        """
        weights = importance_weights or {t: 1.0 for t in self.BIG_FIVE_TRAITS}
        
        vec_a = [traits_profile_a.get(t, 0.5) for t in self.BIG_FIVE_TRAITS]
        vec_b = [traits_profile_b.get(t, 0.5) for t in self.BIG_FIVE_TRAITS]
        w_vals = [weights.get(t, 1.0) for t in self.BIG_FIVE_TRAITS]

        # Weighted Euclidean distance
        weighted_sq_diff = sum(w * ((a - b) ** 2) for a, b, w in zip(vec_a, vec_b, w_vals))
        total_weight = sum(w_vals)
        weighted_dist = math.sqrt(weighted_sq_diff / max(1e-9, total_weight))

        # Cosine similarity
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        cosine_sim = dot / max(1e-9, mag_a * mag_b)

        # Composite affinity score (0 to 100)
        affinity_score = round(max(0.0, min(100.0, (cosine_sim * 0.6 + (1.0 - weighted_dist) * 0.4) * 100.0)), 1)

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "weighted_euclidean_distance": round(weighted_dist, 4),
            "composite_affinity_score": affinity_score,
            "match_grade": "SOULMATE_TIER" if affinity_score >= 85.0 else ("HIGH_COMPATIBILITY" if affinity_score >= 70.0 else ("PASSABLE" if affinity_score >= 50.0 else "LOW_ALIGNMENT")),
            "per_trait_delta": {
                t: round(abs(traits_profile_a.get(t, 0.5) - traits_profile_b.get(t, 0.5)), 2)
                for t in self.BIG_FIVE_TRAITS
            }
        }

    def update_bilateral_elo_ratings(
        self,
        rating_a: float,
        rating_b: float,
        outcome_a: float,
        k_factor: float = 32.0
    ) -> Dict[str, Any]:
        """
        Updates bilateral ELO attractiveness/preference rating.
        outcome_a: 1.0 (A wins/liked), 0.5 (mutual tie/draw), 0.0 (A rejected/loss)
        """
        if rating_a <= 0 or rating_b <= 0:
            raise ValueError("Ratings must be positive.")

        # Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        expected_a = 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a

        outcome_b = 1.0 - outcome_a

        # ELO Delta
        delta_a = round(k_factor * (outcome_a - expected_a), 2)
        delta_b = round(k_factor * (outcome_b - expected_b), 2)

        new_a = round(rating_a + delta_a, 2)
        new_b = round(rating_b + delta_b, 2)

        return {
            "initial_ratings": {"a": rating_a, "b": rating_b},
            "expected_scores": {"a": round(expected_a, 3), "b": round(expected_b, 3)},
            "outcome": {"a": outcome_a, "b": outcome_b},
            "elo_deltas": {"delta_a": delta_a, "delta_b": delta_b},
            "updated_ratings": {"a": new_a, "b": new_b}
        }
