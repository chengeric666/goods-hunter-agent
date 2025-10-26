"""配置模型，集中管理MVP关键参数。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def default_recall_channels() -> List[str]:
    """默认召回通道集合。"""
    return ["behavior_cf", "content_embedding", "category_trending"]


@dataclass
class FeatureConfig:
    """特征配置，包括离线和实时特征清单。"""

    offline_features: List[str] = field(
        default_factory=lambda: [
            "user_click_rate_7d",
            "user_purchase_rate_30d",
            "item_ctr",
            "item_conversion_rate",
            "price_bucket",
            "category_match_score",
        ]
    )
    realtime_features: List[str] = field(
        default_factory=lambda: [
            "current_session_category",
            "current_session_price_band",
            "inventory_status",
        ]
    )


@dataclass
class RecallConfig:
    """召回策略配置。"""

    channels: List[str] = field(default_factory=default_recall_channels)
    channel_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "behavior_cf": 40,
            "content_embedding": 30,
            "category_trending": 30,
        }
    )


@dataclass
class RankingConfig:
    """排序模型配置。"""

    model_name: str = "gbdt_v1"
    exploration_weight: float = 0.1
    feature_importance_threshold: float = 0.01


@dataclass
class ExplorationConfig:
    """探索策略配置。"""

    policy: str = "thompson_sampling"
    exploration_ratio: float = 0.1
    cooldown_clicks: int = 100


@dataclass
class APIConfig:
    """API层配置。"""

    default_page_size: int = 20
    max_page_size: int = 50


@dataclass
class AppConfig:
    """应用级别的总配置。"""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    api: APIConfig = field(default_factory=APIConfig)


DEFAULT_APP_CONFIG = AppConfig()

__all__ = [
    "AppConfig",
    "APIConfig",
    "DEFAULT_APP_CONFIG",
]
