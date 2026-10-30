# -*- coding: utf-8 -*-
"""
GenPark Cyber Dating FastMCP Server
"""

import json
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP
from client import GenParkCyberDating

mcp = FastMCP("genpark-cyber-dating")
dating = GenParkCyberDating()

@mcp.tool()
def solve_gale_shapley_stable_matching(
    proposers_preferences_json: str,
    receivers_preferences_json: str
) -> str:
    """
    Computes stable marriage matching using the Gale-Shapley Deferred Acceptance algorithm.
    """
    proposers = json.loads(proposers_preferences_json)
    receivers = json.loads(receivers_preferences_json)
    res = dating.solve_gale_shapley_stable_matching(proposers, receivers)
    return json.dumps(res, indent=2)

@mcp.tool()
def calculate_semantic_affinity_vector(
    traits_profile_a_json: str,
    traits_profile_b_json: str,
    importance_weights_json: Optional[str] = None
) -> str:
    """
    Calculates cosine similarity and weighted Euclidean distance across Big Five personality traits.
    """
    a = json.loads(traits_profile_a_json)
    b = json.loads(traits_profile_b_json)
    w = json.loads(importance_weights_json) if importance_weights_json else None
    res = dating.calculate_semantic_affinity_vector(a, b, w)
    return json.dumps(res, indent=2)

@mcp.tool()
def update_bilateral_elo_ratings(
    rating_a: float,
    rating_b: float,
    outcome_a: float,
    k_factor: float = 32.0
) -> str:
    """
    Updates bilateral ELO attractiveness/preference rating.
    """
    res = dating.update_bilateral_elo_ratings(rating_a, rating_b, outcome_a, k_factor)
    return json.dumps(res, indent=2)

def main():
    print("Testing Cyber Dating:")
    p = {"A": ["X", "Y"], "B": ["Y", "X"]}
    r = {"X": ["B", "A"], "Y": ["A", "B"]}
    print(solve_gale_shapley_stable_matching(json.dumps(p), json.dumps(r)))

if __name__ == "__main__":
    main()
