"""Where mlsynth's TBRMM hill climb and google/matched_markets' agree, and where
they part company.

Both climb the same objective, so a disagreement is either in the score of a
candidate split -- which would be an arithmetic fault -- or in which candidates
the walk considered, which is a difference between two heuristics and not a
fault in either. The modules here separate those two questions instead of
reporting one number that confounds them.
"""
