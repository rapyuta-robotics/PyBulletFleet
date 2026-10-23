"""Bookkeeping for DataMonitor frames, kept apart from the simulation core.

The core builds a :class:`~pybullet_fleet.gui_commands.MonitorFrame` from its
state and decides *when* by its periodic refresh gate. What lives here is the
rest of the monitor's state:

* the entity list, rebuilt only when an object is added, removed or renamed.
  A scene of thousands of static bodies would otherwise pay for a sorted list
  of every one of them on each refresh. Invalidation drops the cached tuple;
  :class:`~pybullet_fleet.data_monitor.DataMonitor` keys its JSON cache on that
  tuple's identity, so a new tuple is also what renews the JSON;
* whether something the window shows changed since the last frame was built
  (selection, follow, pause, the entity list), and with it the decisions that
  bypass the periodic refresh: a frame while paused, and a flush.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from pybullet_fleet.gui_commands import MonitorEntity, MonitorFrame

if TYPE_CHECKING:
    from pybullet_fleet.sim_object import SimObject


class MonitorFrames:
    """The entity-list cache and the "window is behind" state of one simulation's monitor."""

    def __init__(self) -> None:
        self._entities: Optional[Tuple[MonitorEntity, ...]] = None
        #: Something the window shows changed since the last frame was built.
        self.stale: bool = True
        #: The last frame built, or ``None`` before the first one.
        self.last_frame: Optional[MonitorFrame] = None

    # -- what changed --------------------------------------------------------

    def objects_changed(self) -> None:
        """An object was added, removed or renamed: rebuild the entity list on the next frame."""
        self._entities = None
        self.stale = True

    def view_changed(self) -> None:
        """The selection, follow or pause state changed: the window is behind."""
        self.stale = True

    def reset(self) -> None:
        """A new run starts: no frame yet, and the first one is due."""
        self.last_frame = None
        self.stale = True

    # -- building ------------------------------------------------------------

    def entities(self, objects: Iterable["SimObject"]) -> Tuple[MonitorEntity, ...]:
        """The entity list for ``objects``, reused until :meth:`objects_changed`."""
        if self._entities is None:
            self._entities = tuple(
                MonitorEntity(object_id=obj.object_id, name=obj.name or f"entity-{obj.object_id}", kind=type(obj).__name__)
                for obj in sorted(objects, key=lambda item: item.object_id)
            )
        return self._entities

    def built(self, frame: MonitorFrame) -> None:
        """Record ``frame`` as the latest one: the window is up to date."""
        self.last_frame = frame
        self.stale = False

    # -- decisions outside the periodic refresh --------------------------------

    def due_while_paused(self) -> bool:
        """Whether a paused simulation must build a frame now (nothing refreshes while paused)."""
        return self.stale

    def behind(self, step_count: int) -> bool:
        """Whether the last frame no longer shows the simulation as it is (for a flush)."""
        last = self.last_frame
        return self.stale or last is None or last.steps != step_count
