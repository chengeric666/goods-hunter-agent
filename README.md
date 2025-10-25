# goods-hunter-agent

AI 选品助手 MVP，提供推荐API原型以及多臂老虎机探索策略示例。

## 快速开始

```bash
pip install -e .
uvicorn goods_hunter.api.server:app --reload
```

调用示例：

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_1001", "context_id": "session_electronics", "limit": 5}'
```

## 模块概览

- `docs/mvp_delivery_plan.md`：基于PRD的落地方案。
- `src/goods_hunter/data/feature_store.py`：内存特征中心原型。
- `src/goods_hunter/recall/strategies.py`：多路召回策略及融合逻辑。
- `src/goods_hunter/ranking/gbdt.py`：GBDT风格排序器示例。
- `src/goods_hunter/exploration/mab.py`：Thompson Sampling 与 UCB 探索策略。
- `src/goods_hunter/api/server.py`：FastAPI 服务入口与推荐接口。
