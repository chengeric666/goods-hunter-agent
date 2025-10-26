"""简化版特征中心实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .seed_data import SeedData, default_seed_data


@dataclass
class FeatureView:
    """封装用户、商品及上下文特征。"""

    user_features: Dict[str, float]
    item_features: Dict[str, float]
    context_features: Dict[str, float]

    def merged(self) -> Dict[str, float]:
        """合并所有特征视图，供模型使用。"""

        merged_features: Dict[str, float] = {}
        merged_features.update(self.user_features)
        merged_features.update(self.item_features)
        merged_features.update(self.context_features)
        return merged_features


class FeatureStore:
    """内存内的特征存取示例，MVP阶段主要用作原型演示。"""

    def __init__(self) -> None:
        self._user_features: Dict[str, Dict[str, float]] = {}
        self._item_features: Dict[str, Dict[str, float]] = {}
        self._context_features: Dict[str, Dict[str, float]] = {}

    def batch_write_user_features(self, items: Dict[str, Dict[str, float]]) -> None:
        self._user_features.update(items)

    def batch_write_item_features(self, items: Dict[str, Dict[str, float]]) -> None:
        self._item_features.update(items)

    def batch_write_context_features(self, items: Dict[str, Dict[str, float]]) -> None:
        self._context_features.update(items)

    def get_user_features(self, user_id: str) -> Dict[str, float]:
        return self._user_features.get(user_id, {})

    def get_item_features(self, item_id: str) -> Dict[str, float]:
        return self._item_features.get(item_id, {})

    def get_context_features(self, context_id: Optional[str]) -> Dict[str, float]:
        if context_id is None:
            return {}
        return self._context_features.get(context_id, {})

    def build_feature_view(
        self, user_id: str, item_id: str, context_id: Optional[str]
    ) -> FeatureView:
        """构建特征视图。"""

        return FeatureView(
            user_features=self.get_user_features(user_id),
            item_features=self.get_item_features(item_id),
            context_features=self.get_context_features(context_id),
        )

    def warmup(self, seed_data: SeedData | None = None) -> None:
        """写入样例数据，便于端到端联调。"""

        if self._user_features:
            return
        payload = seed_data or default_seed_data()
        self.batch_write_user_features(payload.user_features)
        self.batch_write_item_features(payload.item_features)
        self.batch_write_context_features(payload.context_features)

    def stream_update(
        self,
        user_id: str,
        feature_updates: Iterable[Dict[str, float]],
    ) -> None:
        """模拟实时特征更新。"""

        current = self._user_features.setdefault(user_id, {})
        for payload in feature_updates:
            current.update(payload)

    def stream_update_item(
        self, item_id: str, feature_updates: Iterable[Dict[str, float]]
    ) -> None:
        """模拟实时商品特征更新。"""

        current = self._item_features.setdefault(item_id, {})
        for payload in feature_updates:
            current.update(payload)

    def stream_update_context(
        self, context_id: str, feature_updates: Iterable[Dict[str, float]]
    ) -> None:
        """模拟实时上下文特征更新。"""

        current = self._context_features.setdefault(context_id, {})
        for payload in feature_updates:
            current.update(payload)


__all__ = ["FeatureStore", "FeatureView"]
