"""简单的交互日志存储。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class InteractionEvent:
    """用户与推荐结果的交互事件。"""

    user_id: str
    item_id: str
    context_id: Optional[str]
    clicked: bool
    purchased: bool


class InteractionStore:
    """将交互记录存储在内存中，并提供聚合统计。"""

    def __init__(self) -> None:
        self._events: List[InteractionEvent] = []

    def record(
        self,
        *,
        user_id: str,
        item_id: str,
        context_id: Optional[str],
        clicked: bool,
        purchased: bool,
    ) -> None:
        self._events.append(
            InteractionEvent(
                user_id=user_id,
                item_id=item_id,
                context_id=context_id,
                clicked=clicked,
                purchased=purchased,
            )
        )

    def summary(self) -> Dict[str, int]:
        """输出交互统计。"""

        clicks = sum(1 for event in self._events if event.clicked)
        purchases = sum(1 for event in self._events if event.purchased)
        return {
            "total_events": len(self._events),
            "clicks": clicks,
            "purchases": purchases,
        }


__all__ = ["InteractionStore", "InteractionEvent"]
