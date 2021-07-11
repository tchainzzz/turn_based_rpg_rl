from dataclasses import dataclass


@dataclass
class Item:
    name: str
    type: str  # weapon, armor, item
    move: str
    hp_bonus: int = 0
    mp_bonus: int = 0
    attack_bonus: int = 0
    defense_bonus: int = 0
