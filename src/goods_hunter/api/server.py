"""FastAPI服务入口。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import AppConfig, DEFAULT_APP_CONFIG
from ..data.feature_store import FeatureStore
from ..data.interaction_store import InteractionStore
from ..data.seed_data import SeedData, default_seed_data
from ..exploration.mab import build_policy
from ..ranking.gbdt import RankedItem, build_default_ranker
from ..recall.strategies import RecallEngine, build_default_recall_engine


class RecommendationRequest(BaseModel):
    """推荐请求体。"""

    user_id: str = Field(..., description="用户ID")
    context_id: Optional[str] = Field(None, description="上下文标识，例如会话")
    limit: Optional[int] = Field(None, description="返回商品数量")


class RankedItemResponse(BaseModel):
    """推荐响应的单个商品。"""

    item_id: str
    score: float
    channel: str
    features: Dict[str, float]


class RecommendationResponse(BaseModel):
    """推荐接口响应体。"""

    user_id: str
    context_id: Optional[str]
    limit: int
    items: List[RankedItemResponse]
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """用于更新探索策略与特征的反馈请求。"""

    user_id: str = Field(..., description="用户ID")
    item_id: str = Field(..., description="商品ID")
    context_id: Optional[str] = Field(None, description="上下文标识")
    clicked: bool = Field(False, description="是否点击")
    purchased: bool = Field(False, description="是否购买")


class GoodsHunterApp:
    """封装系统依赖，便于测试与复用。"""

    def __init__(
        self,
        config: AppConfig = DEFAULT_APP_CONFIG,
        seed_data: SeedData | None = None,
    ) -> None:
        self.config = config
        self.seed_data = seed_data or default_seed_data()
        self.feature_store = FeatureStore()
        self.feature_store.warmup(seed_data=self.seed_data)
        self.recall_engine: RecallEngine = build_default_recall_engine(self.seed_data)
        self.ranker = build_default_ranker()
        self.policy = build_policy(
            policy_name=config.exploration.policy,
            exploration_ratio=config.exploration.exploration_ratio,
        )
        self.interactions = InteractionStore()

    def recommend(self, user_id: str, context_id: Optional[str], limit: int) -> List[RankedItem]:
        candidates = self.recall_engine.recall(
            user_id=user_id, limits=self.config.recall.channel_limits
        )
        ranked = self.ranker.rank(
            user_id=user_id,
            context_id=context_id,
            candidates=candidates,
            feature_lookup=self.feature_store.build_feature_view,
        )
        explored = self.policy.select(ranked)
        return explored[:limit]

    def record_feedback(
        self,
        *,
        user_id: str,
        item_id: str,
        context_id: Optional[str],
        clicked: bool,
        purchased: bool,
    ) -> Dict[str, Dict[str, float]]:
        """更新探索策略与基础特征。"""

        self.interactions.record(
            user_id=user_id,
            item_id=item_id,
            context_id=context_id,
            clicked=clicked,
            purchased=purchased,
        )
        self.policy.update(item_id=item_id, clicked=clicked)
        user_updates: List[Dict[str, float]] = []
        item_updates: List[Dict[str, float]] = []

        if clicked:
            current_click_rate = self.feature_store.get_user_features(user_id).get(
                "user_click_rate_7d", 0.0
            )
            user_updates.append({"user_click_rate_7d": min(1.0, current_click_rate + 0.01)})
            current_item_ctr = self.feature_store.get_item_features(item_id).get(
                "item_ctr", 0.0
            )
            item_updates.append({"item_ctr": min(1.0, current_item_ctr + 0.01)})

        if purchased:
            current_purchase_rate = self.feature_store.get_user_features(user_id).get(
                "user_purchase_rate_30d", 0.0
            )
            user_updates.append(
                {"user_purchase_rate_30d": min(1.0, current_purchase_rate + 0.02)}
            )
            current_item_conv = self.feature_store.get_item_features(item_id).get(
                "item_conversion_rate", 0.0
            )
            item_updates.append(
                {"item_conversion_rate": min(1.0, current_item_conv + 0.015)}
            )

        if context_id and (clicked or purchased):
            inventory = self.feature_store.get_context_features(context_id).get(
                "inventory_status", 0.9
            )
            self.feature_store.stream_update_context(
                context_id, [{"inventory_status": max(0.0, inventory - 0.01)}]
            )

        if user_updates:
            self.feature_store.stream_update(user_id, user_updates)
        if item_updates:
            self.feature_store.stream_update_item(item_id, item_updates)

        return {
            "user_features": self.feature_store.get_user_features(user_id),
            "item_features": self.feature_store.get_item_features(item_id),
        }


def create_app(config: AppConfig = DEFAULT_APP_CONFIG) -> FastAPI:
    """构建FastAPI应用。"""

    goods_hunter = GoodsHunterApp(config=config)
    api = FastAPI(title="Goods Hunter API", version="0.1.0")

    @api.post("/api/v1/recommend", response_model=RecommendationResponse)
    def recommend(payload: RecommendationRequest) -> RecommendationResponse:
        if not payload.user_id:
            raise HTTPException(status_code=400, detail="user_id不能为空")
        limit = payload.limit or config.api.default_page_size
        limit = min(limit, config.api.max_page_size)
        items = goods_hunter.recommend(
            user_id=payload.user_id, context_id=payload.context_id, limit=limit
        )
        response_items = [
            RankedItemResponse(
                item_id=item.item_id,
                score=item.score,
                channel=item.channel,
                features=item.features,
            )
            for item in items
        ]
        metadata: Dict[str, Any] = {
            "config": {
                "ranking": asdict(config.ranking),
                "exploration": asdict(config.exploration),
            },
            "selected_policy": config.exploration.policy,
            "interaction_totals": goods_hunter.interactions.summary(),
        }
        return RecommendationResponse(
            user_id=payload.user_id,
            context_id=payload.context_id,
            limit=limit,
            items=response_items,
            metadata=metadata,
        )

    @api.post("/api/v1/feedback")
    def feedback(payload: FeedbackRequest) -> Dict[str, Any]:
        if not payload.user_id or not payload.item_id:
            raise HTTPException(status_code=400, detail="user_id与item_id不能为空")
        updated_features = goods_hunter.record_feedback(
            user_id=payload.user_id,
            item_id=payload.item_id,
            context_id=payload.context_id,
            clicked=payload.clicked,
            purchased=payload.purchased,
        )
        return {
            "message": "feedback recorded",
            "updated_features": updated_features,
            "interaction_totals": goods_hunter.interactions.summary(),
        }

    return api


app = create_app()

__all__ = [
    "create_app",
    "app",
    "GoodsHunterApp",
    "RecommendationRequest",
    "RecommendationResponse",
    "FeedbackRequest",
]
