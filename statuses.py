from dataclasses import dataclass
from turn_based_env import Agent

STATUS_CHECK_INTERVAL = 6


@dataclass
class StatusManager:
    name: str
    source: Agent
    target: Agent
    time_applied: int
    check_every: int = STATUS_CHECK_INTERVAL
