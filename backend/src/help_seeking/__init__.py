"""Learning when and whom to ask for help under changing teammate reliability."""
__version__ = "1.0.0"

# Core modules (original single-requester)
from . import algorithms, baselines, environment, experiments, metrics, policies, reliability, simulation, stats, trust, utilities

# Multi-agent modules (new)
from . import autonomous_agent, acceptance, helper_qlearning, multiagent_policies, multiagent_environment, multiagent_simulation

__all__ = [
    # Original
    "algorithms", "baselines", "environment", "experiments", "metrics",
    "policies", "reliability", "simulation", "stats", "trust", "utilities",
    # New multi-agent
    "autonomous_agent", "acceptance", "helper_qlearning",
    "multiagent_policies", "multiagent_environment", "multiagent_simulation",
]