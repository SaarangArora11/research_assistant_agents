"""Event bus for asynchronous agent events and live Server-Sent Events (SSE)."""

from __future__ import annotations
import asyncio
import json
from typing import Dict, List, Optional, Set
from app.models import AgentEvent


class EventBus:
    """In-memory pub-sub event bus for streaming agent activity to web clients."""

    def __init__(self, history_limit: int = 500):
        self._subscribers: Set[asyncio.Queue] = set()
        self._run_subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._history: Dict[str, List[AgentEvent]] = {}
        self._history_limit = history_limit
        self._lock = asyncio.Lock()

    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to all subscribers and append to run history."""
        async with self._lock:
            # Store in history
            if event.run_id not in self._history:
                self._history[event.run_id] = []
            self._history[event.run_id].append(event)
            if len(self._history[event.run_id]) > self._history_limit:
                self._history[event.run_id].pop(0)

            # Broadcast to global subscribers
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

            # Broadcast to run-specific subscribers
            if event.run_id in self._run_subscribers:
                for q in list(self._run_subscribers[event.run_id]):
                    try:
                        q.put_nowait(event)
                    except Exception:
                        pass

    def get_history(self, run_id: str) -> List[AgentEvent]:
        """Retrieve recent events for a specific run."""
        return list(self._history.get(run_id, []))

    async def subscribe(self, run_id: Optional[str] = None) -> asyncio.Queue:
        """Subscribe to events, optionally filtered by run_id."""
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if run_id:
                if run_id not in self._run_subscribers:
                    self._run_subscribers[run_id] = set()
                self._run_subscribers[run_id].add(q)
            else:
                self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue, run_id: Optional[str] = None) -> None:
        """Remove a subscriber queue."""
        async with self._lock:
            if run_id and run_id in self._run_subscribers:
                self._run_subscribers[run_id].discard(q)
                if not self._run_subscribers[run_id]:
                    del self._run_subscribers[run_id]
            else:
                self._subscribers.discard(q)


# Global singleton instance
event_bus = EventBus()
