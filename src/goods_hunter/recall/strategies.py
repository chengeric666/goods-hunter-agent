"""召回策略组件。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..data.seed_data import SeedData, default_seed_data


@dataclass
class ItemCandidate:
    """召回候选商品。"""

    item_id: str
    channel: str
    score: float


class RecallStrategy:
    """召回策略基类。"""

    channel: str

    def recall(self, user_id: str, limit: int) -> Sequence[ItemCandidate]:
        raise NotImplementedError


class BehaviorCFRecall(RecallStrategy):
    """行为协同过滤召回。"""

    channel = "behavior_cf"

    def __init__(self, co_click_matrix: Dict[str, Dict[str, float]]) -> None:
        self._co_click_matrix = co_click_matrix

    def recall(self, user_id: str, limit: int) -> Sequence[ItemCandidate]:
        user_items = self._co_click_matrix.get(user_id, {})
        sorted_items = sorted(user_items.items(), key=lambda kv: kv[1], reverse=True)
        return [
            ItemCandidate(item_id=item_id, channel=self.channel, score=score)
            for item_id, score in sorted_items[:limit]
        ]


class ContentEmbeddingRecall(RecallStrategy):
    """内容向量相似度召回。"""

    channel = "content_embedding"

    def __init__(self, anchor_vector: Dict[str, float]) -> None:
        self._anchor_vector = anchor_vector

    def recall(self, user_id: str, limit: int) -> Sequence[ItemCandidate]:
        sorted_items = sorted(
            self._anchor_vector.items(), key=lambda kv: kv[1], reverse=True
        )
        return [
            ItemCandidate(item_id=item_id, channel=self.channel, score=score)
            for item_id, score in sorted_items[:limit]
        ]


class CategoryTrendingRecall(RecallStrategy):
    """品类热度召回。"""

    channel = "category_trending"

    def __init__(self, category_top_items: Dict[str, List[str]]) -> None:
        self._category_top_items = category_top_items

    def recall(self, user_id: str, limit: int) -> Sequence[ItemCandidate]:
        items: List[ItemCandidate] = []
        for category, sku_list in self._category_top_items.items():
            for idx, sku in enumerate(sku_list[:limit]):
                items.append(
                    ItemCandidate(
                        item_id=sku,
                        channel=self.channel,
                        score=max(0.0, 1.0 - idx * 0.05),
                    )
                )
        return items[:limit]


class RecallEngine:
    """聚合多路召回策略。"""

    def __init__(self, strategies: Iterable[RecallStrategy]) -> None:
        self._strategies = list(strategies)

    def recall(self, user_id: str, limits: Dict[str, int]) -> List[ItemCandidate]:
        candidates: List[ItemCandidate] = []
        for strategy in self._strategies:
            limit = limits.get(strategy.channel, 20)
            candidates.extend(strategy.recall(user_id=user_id, limit=limit))
        deduplicated: Dict[str, ItemCandidate] = {}
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            deduplicated.setdefault(candidate.item_id, candidate)
        return list(deduplicated.values())


def build_default_recall_engine(seed_data: SeedData | None = None) -> RecallEngine:
    """创建默认召回引擎。"""

    payload = seed_data or default_seed_data()
    cf = BehaviorCFRecall(co_click_matrix=payload.co_click_matrix)
    embedding = ContentEmbeddingRecall(anchor_vector=payload.content_embedding)
    trending = CategoryTrendingRecall(category_top_items=payload.category_top_items)
    return RecallEngine([cf, embedding, trending])


__all__ = [
    "ItemCandidate",
    "RecallEngine",
    "build_default_recall_engine",
]
