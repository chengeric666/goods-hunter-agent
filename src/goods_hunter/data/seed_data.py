"""内置样例数据，方便MVP阶段快速初始化。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SeedData:
    """封装原型所需的全部静态数据。"""

    user_features: Dict[str, Dict[str, float]]
    item_features: Dict[str, Dict[str, float]]
    context_features: Dict[str, Dict[str, float]]
    co_click_matrix: Dict[str, Dict[str, float]]
    content_embedding: Dict[str, float]
    category_top_items: Dict[str, List[str]]


def default_seed_data() -> SeedData:
    """返回默认的演示数据。"""

    user_features = {
        "u_1001": {
            "user_click_rate_7d": 0.35,
            "user_purchase_rate_30d": 0.08,
            "preferred_category_electronics": 1.0,
            "preferred_price_low": 0.4,
            "preferred_price_high": 0.1,
        },
        "u_1002": {
            "user_click_rate_7d": 0.2,
            "user_purchase_rate_30d": 0.03,
            "preferred_category_beauty": 1.0,
            "preferred_price_mid": 0.6,
        },
        "u_1003": {
            "user_click_rate_7d": 0.28,
            "user_purchase_rate_30d": 0.05,
            "preferred_category_home": 1.0,
            "preferred_price_mid": 0.5,
            "preferred_price_low": 0.3,
        },
    }

    item_features = {
        "sku_2001": {
            "item_ctr": 0.18,
            "item_conversion_rate": 0.05,
            "price_bucket": 2.0,
            "category_match_score": 0.8,
            "logistics_score": 0.7,
        },
        "sku_2002": {
            "item_ctr": 0.12,
            "item_conversion_rate": 0.07,
            "price_bucket": 3.0,
            "category_match_score": 0.9,
            "logistics_score": 0.8,
        },
        "sku_2003": {
            "item_ctr": 0.08,
            "item_conversion_rate": 0.02,
            "price_bucket": 1.0,
            "category_match_score": 0.6,
            "logistics_score": 0.6,
        },
        "sku_2004": {
            "item_ctr": 0.16,
            "item_conversion_rate": 0.04,
            "price_bucket": 2.0,
            "category_match_score": 0.7,
            "logistics_score": 0.75,
        },
        "sku_2005": {
            "item_ctr": 0.14,
            "item_conversion_rate": 0.06,
            "price_bucket": 4.0,
            "category_match_score": 0.85,
            "logistics_score": 0.82,
        },
    }

    context_features = {
        "session_electronics": {
            "current_session_category": 1.0,
            "current_session_price_band": 2.0,
            "inventory_status": 0.9,
        },
        "session_beauty": {
            "current_session_category": 0.5,
            "current_session_price_band": 3.0,
            "inventory_status": 0.95,
        },
        "session_home": {
            "current_session_category": 0.8,
            "current_session_price_band": 1.5,
            "inventory_status": 0.92,
        },
    }

    co_click_matrix = {
        "u_1001": {"sku_2001": 0.9, "sku_2002": 0.6, "sku_2005": 0.55},
        "u_1002": {"sku_2002": 0.7, "sku_2003": 0.5, "sku_2004": 0.45},
        "u_1003": {"sku_2003": 0.6, "sku_2004": 0.58, "sku_2001": 0.4},
    }

    content_embedding = {
        "sku_2001": 0.85,
        "sku_2002": 0.75,
        "sku_2003": 0.6,
        "sku_2004": 0.65,
        "sku_2005": 0.7,
    }

    category_top_items = {
        "electronics": ["sku_2001", "sku_2002", "sku_2005"],
        "beauty": ["sku_2003", "sku_2004"],
        "home": ["sku_2004", "sku_2001"],
    }

    return SeedData(
        user_features=user_features,
        item_features=item_features,
        context_features=context_features,
        co_click_matrix=co_click_matrix,
        content_embedding=content_embedding,
        category_top_items=category_top_items,
    )


__all__ = ["SeedData", "default_seed_data"]
