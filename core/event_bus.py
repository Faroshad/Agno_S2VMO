#!/usr/bin/env python3
"""
Lightweight in-process event bus for agent coordination.

This keeps infrastructure minimal while allowing simulation and chat agents
in the same process to share state changes in near real time.
"""

from dataclasses import dataclass
from queue import Queue, Empty
from threading import Lock
from typing import List


@dataclass(frozen=True)
class SimulationCycleEvent:
    """Event published when a simulation cycle completes."""

    cycle: int
    timestamp: str
    voxels_updated: int
    max_stress: float
    avg_stress: float
    fem_skipped: bool = False


class EventBus:
    """Thread-safe queue based event bus for internal coordination."""

    def __init__(self) -> None:
        self._queue: Queue = Queue()
        self._lock = Lock()

    def publish_cycle_complete(self, event: SimulationCycleEvent) -> None:
        with self._lock:
            self._queue.put(event)

    def drain_cycle_events(self) -> List[SimulationCycleEvent]:
        """
        Drain all pending cycle-complete events.

        Returns:
            List of events in FIFO order.
        """
        events: List[SimulationCycleEvent] = []
        while True:
            try:
                event = self._queue.get_nowait()
                if isinstance(event, SimulationCycleEvent):
                    events.append(event)
            except Empty:
                break
        return events


event_bus = EventBus()


__all__ = ["SimulationCycleEvent", "EventBus", "event_bus"]
