from dataclasses import dataclass, field
import random
import string
from typing import List, Tuple

import numpy as np

from base_logger import logger
from interactibles import Item

PARTY_DEATH_PENALTY = 0

ID_LENGTH = 8

@dataclass
class Agent:
    name: str
    position: int
    hp: int
    mp: int
    attack: int
    defense: int = 0
    statuses: List[Tuple[str, int]] = field(default_factory=list) # list of (status_name, time_applied)
    status_resist: float = 0
    special_moves: List[str] = field(default_factory=list)
    has_gone: bool = False

    def __post_init__(self):
        self.max_hp = self.hp
        self.max_mp = self.mp
        self.special_moves = self.special_moves.strip().split()
        self.id = str(self.position) + '_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=ID_LENGTH))

    def execute(self, state, action, target_list):
        reward = 0
        self.hp = np.clip(self.hp - action.hp_cost, 0, self.max_hp)
        self.mp = np.clip(self.mp - action.mp_cost, 0, self.max_mp)
        self.attack = max(self.attack - action.attack_cost, 0)
        self.defense = max(self.defense - action.defense_cost, 0)
        if random.random() < action.status_self_p:
            self.status = action.status_self
        for target in target_list:
            target.hp = np.clip(target.hp + action.hp_delta, 0, target.max_hp)
            if target.hp == 0:
                if isinstance(target, Ally):
                    logger.info(f"KILL: PARTY {target.name} (id: {target.id}) (-{-PARTY_DEATH_PENALTY})")
                    state.party.remove(target)
                    reward += PARTY_DEATH_PENALTY
                else: # Enemy
                    logger.info(f"KILL: ENEMY {target.name} (id: {target.id}) (+{target.gold})")
                    state.enemies.remove(target)
                    state.gold += target.gold
                    reward += target.gold
                break
            target.mp = np.clip(target.mp + action.mp_delta, 0, target.max_mp)
            target.attack = max(target.attack + action.attack_delta, 0)
            target.defense = max(target.defense + action.defense_delta, 0)
            if random.random() < action.status_target_p:
                target.status = action.status_target
        state = self.apply_status_effects(state)

        self.has_gone = True
        state.turns_left -= 1
        state.global_step += 1
        return state, reward

    def apply_status_effects(self, state):
        for agent in state.party + state.enemies:
            for status in agent.statuses:
                if status.check_needed(state.global_step):
                    status.tick()
                    if status.can_escape():
                        agent.statuses.remove(status)
        return state

    def get_legal_actions(self):
        action_names = ['basic'] + self.special_moves
        return action_names

    def __eq__(self, other):
        return self is other


@dataclass
class Ally(Agent):
    weapon: Item = None
    armor: Item = None
    items: List[Item] = field(default_factory=list)

    def equip(self, item: Item):
        self.attack += item.attack_bonus
        self.hp += item.hp_bonus
        self.mp += item.mp_bonus

    def get_legal_actions(self):
        action_names = ['basic'] + self.special_moves + [item.move for item in self.items]
        return action_names


@dataclass
class Enemy(Agent):
    min_level: int = 0
    max_level: int = 0
    gold: int = 1

