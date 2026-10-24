"""Bounded in-process pub/sub, marshalled onto the ASGI event loop."""

import asyncio
from typing import Set, Dict, Any, List
import logging

logger = logging.getLogger("VisionShield.EventHub")


class EventHub:
    def __init__(self):
        # Active client queues for real-time WebSocket fanout
        self._subscribers: Set[asyncio.Queue] = set()
        self._recent_events: List[Dict[str, Any]] = []
        self._max_recent = 100
        self._loop = None

    def bind(self, loop):
        self._loop = loop
        if loop is None:
            self._subscribers.clear()
            self._recent_events.clear()

    def publish(self, event):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event):
        asyncio.create_task(self.broadcast(event))

    def subscribe(self) -> asyncio.Queue:
        """Register a new client subscription queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a client subscription queue."""
        self._subscribers.discard(queue)

    async def broadcast(self, event: Dict[str, Any]):
        """Broadcast an event payload to all active WebSocket listeners."""
        # Keep recent event in ring buffer for immediate dashboard hydration
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_recent:
            self._recent_events.pop(0)

        # Distribute to all connected queues
        dead_queues = []
        for queue in self._subscribers:
            try:
                if queue.full():
                    try:
                        queue.get_nowait()  # Drop oldest if congested
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(event)
            except Exception as e:
                logger.warning(f"Error publishing to client queue: {e}")
                dead_queues.append(queue)

        for dead in dead_queues:
            self._subscribers.discard(dead)

    def get_recent_events(self) -> List[Dict[str, Any]]:
        """Return the most recent events."""
        return list(self._recent_events)


# Global singleton event hub
event_hub = EventHub()
