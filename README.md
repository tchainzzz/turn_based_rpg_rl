# Turn-Based RPG Reinforcement Learning

Goal: a combat system for RPGs, designed for benchmarking reinforcement learning algorithms.

Only very basic combat code (MVP) is finished.

This project is licensed under LGPL-3.0.

## Motivation

I was playing a turn based RPG and really wanted to figure out the optimal strategy. It bothered me. **Note: will write more rigorous motivation later.**

## Setup

### State and Action Space

We consider a Party vs. Party setup. One party is controlled by the player; a second party is controlled by an AI. Each party is a list of Agents, individual entities that can act. 

Each Agent has a set of statistics associated with it; for example:
* HP: how much damage an Agent can take before it "dies" (i.e. is removed from the party)
* Attack: how much damage an Agent can deal to another agent.

For a full list, consult the `Agent`, `Ally`, and `Enemy` classes in `turn_based_env.py`.

The state space consists of all of the statistics of each agent in the aggregate, as well as a few pieces of metadata, such as the "level" (i.e. # of parties faced).

The action space can be represented in two parts: 1) deltas to agent statistics, and 2) a list of target agents. When an action is applied, the agents specified in the target list have the deltas applied. 

### Rewards

Each Party's objective is to eliminate all of the opposing party members with their attacks. When the player party count reaches 0, the game is over (terminal state). When the enemy party count reaches 0, the player "beats" the level, and a new level is generated with new enemies.

Upon defeating each individual enemy in a party, a small reward ("gold") is given.

Currently, as with many such video games, the AI system is pretty rudimentary, selecting a random attack.

### Agents

Agents can either be allies or enemies. Their statistics are specified in the CSV files `allies.csv` and `enemies.csv`, respectively. 

Allies can possibly equip items as well, such as weapons, armor, or auxilliary combat items. Each of these items either 1) grants the agent a statistical bonus (weapons, armor, items), or 2) grants the agent an additional action type (i.e. healing, "spells").

## Machine Learning Modeling

Forthcoming! Based on the OpenAI Five Actor-Critic style approach, the actor network will choose an action and a target, with the critic supervised by the reward signal. The current reward function is simply gold, but there is functionality for incorporating levels and actions-per-level in the future.
