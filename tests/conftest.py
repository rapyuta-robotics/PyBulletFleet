"""Shared fixtures for all test modules."""

from contextlib import contextmanager
from types import SimpleNamespace

import pybullet as p
import pybullet_data
import pytest

from pybullet_fleet.data_monitor import DataMonitor
from pybullet_fleet.events import EventBus
from pybullet_fleet.sim_object import SimObject


# ---------------------------------------------------------------------------
# Shared mock sim_core classes
# ---------------------------------------------------------------------------


class MockSimCore:
    """Minimal sim_core mock for tests.

    Provides object registry, time tracking, and optional physics stepping.

    When ``physics=True``, ``tick()`` calls ``p.stepSimulation()`` each step
    so that motor-controlled joints (via ``setJointMotorControl2``) take effect.

    When ``physics=False`` (default), ``tick()`` only advances ``sim_time``
    — joints move via ``resetJointState`` (kinematic interpolation).
    """

    def __init__(self, dt: float = 1.0 / 240.0, *, physics: bool = False):
        self.sim_time: float = 0.0
        self._dt: float = dt
        self.sim_objects: list = []
        self._next_object_id: int = 0
        self._kinematic_objects: set = set()
        self._client: int = 0
        self._params = SimpleNamespace(physics=physics)
        self.events: EventBus = EventBus()
        # Two-phase step state — mock stays in "outside step_once" mode so all
        # set_pose calls take the immediate path (legacy behaviour for tests).
        self._in_step: bool = False
        self._pending_pose_ids: set = set()
        self._link_attachment_parent_ids: set[int] = set()

    @property
    def client(self) -> int:
        return self._client

    def add_object(self, obj):
        self.sim_objects.append(obj)

    def _record_state_spawn(self, obj, spawn_params):
        """Match the core recording hook; the test double does not record."""
        return None

    def remove_object(self, obj):
        if obj in self.sim_objects:
            self.sim_objects.remove(obj)

    def _mark_object_moved(self, object_id):
        pass

    def _register_link_attachment_parent(self, object_id: int) -> None:
        self._link_attachment_parent_ids.add(object_id)

    def _unregister_link_attachment_parent(self, object_id: int) -> None:
        self._link_attachment_parent_ids.discard(object_id)

    @contextmanager
    def batch_spawn(self):
        """No-op batch_spawn context for tests (mirrors MultiRobotSimulationCore)."""
        yield

    def register_manager(self, manager) -> None:
        """No-op stub — AgentManager registers itself on construction."""
        if not hasattr(self, "_registered_managers"):
            self._registered_managers: list = []
        if manager not in self._registered_managers:
            self._registered_managers.append(manager)
            manager.sim_core = self

    def register_callback(self, callback, frequency=None):
        """Record registered callbacks (used by AgentManager tests)."""
        if not hasattr(self, "_registered_callbacks"):
            self._registered_callbacks: list = []
        self._registered_callbacks.append({"func": callback, "frequency": frequency})

    def tick(self, n: int = 1):
        """Advance sim_time by *n* time-steps, with physics stepping if enabled.

        Also calls ``obj.update(dt)`` for every registered object that has
        ``_needs_update = True``, mirroring the unified loop in
        ``MultiRobotSimulationCore.step_once()``.
        """
        for _ in range(n):
            if self._params.physics:
                p.stepSimulation(physicsClientId=self._client)
            self.sim_time += self._dt
            for obj in self.sim_objects:
                if getattr(obj, "_needs_update", False):
                    obj.update(self._dt)


@pytest.fixture(autouse=True)
def _clear_shared_shapes():
    """Clear SimObject shape cache before every test.

    ``SimObject._shared_shapes`` is a class-level dict that caches PyBullet
    shape IDs.  Those IDs become invalid after ``p.disconnect()``, so they
    must be cleared before each test to prevent stale IDs from being reused
    in a new physics session.
    """
    SimObject._shared_shapes.clear()


@pytest.fixture(autouse=True)
def _disable_monitor_gui(monkeypatch):
    """Prevent DataMonitor from opening matplotlib windows during tests.

    Even when ``monitor=False`` is passed to SimulationParams, some code
    paths may still instantiate a DataMonitor.  This fixture globally
    patches ``DataMonitor.start`` to be a no-op so that matplotlib windows
    never appear during headless test runs.
    """
    monkeypatch.setattr(DataMonitor, "start", lambda self: None)


# ---------------------------------------------------------------------------
# Shared PyBullet fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reject_mode(monkeypatch):
    """Make ``goal_replacement="reject"`` the default, for tests of refused goals."""
    from pybullet_fleet import _defaults

    monkeypatch.setitem(_defaults.CONTROLLER, "goal_replacement", "reject")


@pytest.fixture
def pybullet_env():
    """Headless PyBullet DIRECT session with ground plane.

    Provides a physics client ID usable by any test that needs a real
    PyBullet environment.  The session is torn down after each test.
    """
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -10, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    yield client
    p.disconnect(client)


@pytest.fixture(
    params=[
        pytest.param("physics", id="physics"),
        pytest.param("kinematic", id="kinematic"),
        pytest.param("physics_off", id="physics_off"),
    ]
)
def arm_sim(request, pybullet_env):
    """Parametrized (sim_core, mass) for arm robot tests.

    Provides three simulation modes — tests pair this with any URDF via
    their own ``create_*_agent()`` helper:

    - **physics**: ``mass=None`` (URDF values) + ``stepSimulation`` each tick.
    - **kinematic**: ``mass=0.0`` + no ``stepSimulation`` (pure kinematic).
    - **physics_off**: ``mass=None`` + no ``stepSimulation``
      (agent auto-detects kinematic interpolation).
    """
    if request.param == "physics":
        sc = MockSimCore(physics=True)
    else:
        sc = MockSimCore(physics=False)
    sc._client = pybullet_env
    if request.param == "kinematic":
        return sc, 0.0
    return sc, None
