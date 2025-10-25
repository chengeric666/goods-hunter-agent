"""FastAPI服务入口。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import AppConfig, DEFAULT_APP_CONFIG
from ..data.feature_store import FeatureStore
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


class GoodsHunterApp:
    """封装系统依赖，便于测试与复用。"""

    def __init__(self, config: AppConfig = DEFAULT_APP_CONFIG) -> None:
        self.config = config
        self.feature_store = FeatureStore()
        self.feature_store.warmup()
        self.recall_engine: RecallEngine = build_default_recall_engine()
        self.ranker = build_default_ranker()
        self.policy = build_policy(
            policy_name=config.exploration.policy,
            exploration_ratio=config.exploration.exploration_ratio,
        )

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
        }
        return RecommendationResponse(
            user_id=payload.user_id,
            context_id=payload.context_id,
            limit=limit,
            items=response_items,
            metadata=metadata,
        )

    return api


app = create_app()

__all__ = ["create_app", "app", "GoodsHunterApp"]
