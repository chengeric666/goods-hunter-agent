"""多臂老虎机策略。"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

from ..ranking.gbdt import RankedItem


@dataclass
class ArmState:
    """记录臂的历史表现。"""

    impressions: int = 0
    clicks: int = 0

    @property
    def ctr(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions


@dataclass
class ThompsonSamplingPolicy:
    """Beta-Bernoulli的Thompson Sampling策略。"""

    exploration_ratio: float
    states: Dict[str, ArmState] = field(default_factory=dict)

    def select(self, items: List[RankedItem]) -> List[RankedItem]:
        selected: List[RankedItem] = []
        for item in items:
            state = self.states.setdefault(item.item_id, ArmState())
            alpha = 1 + state.clicks
            beta = 1 + state.impressions - state.clicks
            sample = random.betavariate(alpha, beta)
            adjusted_score = (
                (1 - self.exploration_ratio) * item.score
                + self.exploration_ratio * sample
            )
            selected.append(
                RankedItem(
                    item_id=item.item_id,
                    score=adjusted_score,
                    channel=item.channel,
                    features=item.features,
                )
            )
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected

    def update(self, item_id: str, clicked: bool) -> None:
        state = self.states.setdefault(item_id, ArmState())
        state.impressions += 1
        if clicked:
            state.clicks += 1


@dataclass
class UCBPolicy:
    """上置信界（UCB1）策略。"""

    exploration_ratio: float
    states: Dict[str, ArmState] = field(default_factory=dict)

    def select(self, items: List[RankedItem]) -> List[RankedItem]:
        total_impressions = sum(state.impressions for state in self.states.values()) + 1
        selected: List[RankedItem] = []
        for item in items:
            state = self.states.setdefault(item.item_id, ArmState())
            if state.impressions == 0:
                bonus = 1.0
            else:
                bonus = math.sqrt((2 * math.log(total_impressions)) / state.impressions)
            adjusted_score = item.score + self.exploration_ratio * bonus
            selected.append(
                RankedItem(
                    item_id=item.item_id,
                    score=adjusted_score,
                    channel=item.channel,
                    features=item.features,
                )
            )
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected

    def update(self, item_id: str, clicked: bool) -> None:
        state = self.states.setdefault(item_id, ArmState())
        state.impressions += 1
        if clicked:
            state.clicks += 1


def build_policy(policy_name: str, exploration_ratio: float):
    """根据名称构造策略实例。"""

    if policy_name == "thompson_sampling":
        return ThompsonSamplingPolicy(exploration_ratio=exploration_ratio)
    if policy_name == "ucb":
        return UCBPolicy(exploration_ratio=exploration_ratio)
    raise ValueError(f"未知策略: {policy_name}")


__all__ = [
    "ArmState",
    "ThompsonSamplingPolicy",
    "UCBPolicy",
    "build_policy",
]
