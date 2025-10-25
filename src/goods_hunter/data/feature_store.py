"""简化版特征中心实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


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

    def warmup(self) -> None:
        """写入样例数据，便于端到端联调。"""

        if self._user_features:
            return
        self.batch_write_user_features(
            {
                "u_1001": {
                    "user_click_rate_7d": 0.35,
                    "user_purchase_rate_30d": 0.08,
                    "preferred_category_electronics": 1.0,
                },
                "u_1002": {
                    "user_click_rate_7d": 0.2,
                    "user_purchase_rate_30d": 0.03,
                    "preferred_category_beauty": 1.0,
                },
            }
        )
        self.batch_write_item_features(
            {
                "sku_2001": {
                    "item_ctr": 0.18,
                    "item_conversion_rate": 0.05,
                    "price_bucket": 2.0,
                    "category_match_score": 0.8,
                },
                "sku_2002": {
                    "item_ctr": 0.12,
                    "item_conversion_rate": 0.07,
                    "price_bucket": 3.0,
                    "category_match_score": 0.9,
                },
                "sku_2003": {
                    "item_ctr": 0.08,
                    "item_conversion_rate": 0.02,
                    "price_bucket": 1.0,
                    "category_match_score": 0.6,
                },
            }
        )
        self.batch_write_context_features(
            {
                "session_electronics": {
                    "current_session_category": 1.0,
                    "current_session_price_band": 2.0,
                    "inventory_status": 0.9,
                }
            }
        )

    def stream_update(
        self,
        user_id: str,
        feature_updates: Iterable[Dict[str, float]],
    ) -> None:
        """模拟实时特征更新。"""

        current = self._user_features.setdefault(user_id, {})
        for payload in feature_updates:
            current.update(payload)


__all__ = ["FeatureStore", "FeatureView"]
