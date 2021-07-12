from dataclasses import dataclass, field
import math
import random
import string
from typing import List, Tuple

import numpy as np
import pandas as pd

from base_logger import logger
from interactibles import Item


FALLBACK_ATTACK_NAME = 'basic'
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
        self.id = f"{self.name}@{self.position}"

    def short_repr(self):
        repr_str = ""
        if isinstance(self, Ally):
            repr_str += "PARTY"
        else:
            repr_str += "ENEMY"
        repr_str += f" {self.name} (id: {self.id})"
        return repr_str

    def effect_scale(self, target_field: str, action, action_field: str):
        if action.effect_scaling == 'proportional':
            return getattr(self, target_field) * getattr(action, action_field)
        else:
            return getattr(action, action_field)

    def execute(self, state, action, target_list):
        target_ids = [target.id for target in target_list]
        logger.debug(f"AGENT {self.id} --({action.name})-> AGENT {target_ids}\nParameters:{action.pretty_repr()}")
        reward = 0
        effective_hp_cost = self.effect_scale('hp', action, 'hp_cost')
        effective_mp_cost = self.effect_scale('mp', action, 'mp_cost')
        effective_attack_cost = self.effect_scale('attack', action, 'attack_cost')
        effective_defense_cost = self.effect_scale('defense', action, 'defense_cost')
        effective_hp_delta = self.effect_scale('attack', action, 'hp_delta')

        # TODO: effective_mp/atk/def_delta seems very niche, probably not going to implement now

        self.hp = np.clip(self.hp - effective_hp_cost, 0, self.max_hp)
        self.mp = np.clip(self.mp - effective_mp_cost, 0, self.max_mp)
        self.attack = max(self.attack - effective_attack_cost, 0)
        self.defense = max(self.defense - effective_defense_cost, 0)
        if random.random() < action.status_self_p:
            self.status = action.status_self
        for target in target_list:
            target.hp = np.clip(target.hp + effective_hp_delta, 0, target.max_hp)
            if target.hp == 0:
                info_str = target.short_repr()
                if isinstance(target, Ally):
                    logger.info(f"KILL: {info_str} (-{-PARTY_DEATH_PENALTY})")
                    state.party.remove(target)
                    reward += PARTY_DEATH_PENALTY
                else: # Enemy
                    logger.info(f"KILL: {info_str} (+{target.gold})")
                    state.enemies.remove(target)
                    state.gold += target.gold
                    reward += target.gold
                continue
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

    def get_candidate_moves(self):
        return self.special_moves

    def get_legal_actions(self, entity_bank):
        action_names = [FALLBACK_ATTACK_NAME]
        for move_name in self.get_candidate_moves():
            if self.mp >= entity_bank.action_data.loc[move_name].get('mp_cost'):
                action_names.append(move_name)
        info_str = self.short_repr()
        logger.debug(f"{info_str} ACTIONS: {action_names}")
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
        self.max_hp += item.hp_bonus
        self.max_mp += item.mp_bonus
        self.items.append(item)

    def get_candidate_moves(self):
        return self.special_moves + [item.move for item in self.items if not pd.isnull(item.move)]


@dataclass
class Enemy(Agent):
    min_level: int = 0
    max_level: int = 0
    gold: int = 1

