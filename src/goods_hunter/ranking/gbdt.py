"""排序模型模块，模拟GBDT打分逻辑。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..recall.strategies import ItemCandidate


@dataclass
class RankedItem:
    """排序后的商品结果。"""

    item_id: str
    score: float
    channel: str
    features: Dict[str, float]


class GBDTRanker:
    """使用手工设定的特征权重模拟GBDT推理。"""

    def __init__(self, feature_weights: Dict[str, float]) -> None:
        self._feature_weights = feature_weights

    def score(self, features: Dict[str, float]) -> float:
        return sum(features.get(name, 0.0) * weight for name, weight in self._feature_weights.items())

    def rank(
        self,
        user_id: str,
        context_id: str | None,
        candidates: Iterable[ItemCandidate],
        feature_lookup,
    ) -> List[RankedItem]:
        ranked: List[RankedItem] = []
        for candidate in candidates:
            feature_view = feature_lookup(user_id, candidate.item_id, context_id)
            merged_features = feature_view.merged()
            score = self.score(merged_features)
            ranked.append(
                RankedItem(
                    item_id=candidate.item_id,
                    score=score,
                    channel=candidate.channel,
                    features=merged_features,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked


def build_default_ranker() -> GBDTRanker:
    """创建默认GBDT排序器。"""

    return GBDTRanker(
        feature_weights={
            "user_click_rate_7d": 0.25,
            "user_purchase_rate_30d": 0.4,
            "item_ctr": 0.2,
            "item_conversion_rate": 0.3,
            "category_match_score": 0.1,
            "preferred_category_electronics": 0.15,
            "preferred_category_beauty": 0.12,
            "inventory_status": 0.05,
        }
    )


__all__ = ["GBDTRanker", "RankedItem", "build_default_ranker"]
