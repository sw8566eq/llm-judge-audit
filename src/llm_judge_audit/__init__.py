"""llm_judge_audit — a statistical audit of LLM-as-judge reliability.

Submodules:
    judges              LLM judge orchestration across prompt variants.
    pipeline            Orchestrates judges output into classical/causal/Bayesian summaries + dashboard JSON.
    classical_stats      Inter-rater reliability, calibration, McNemar's test.
    causal_experiment    Within-item paired/crossed experiment on judge-prompt design.
    bayesian_model       Hierarchical (Rasch/IRT-style) model of judge bias and item difficulty.
    viz                  Plotting / dashboard data prep.
"""
