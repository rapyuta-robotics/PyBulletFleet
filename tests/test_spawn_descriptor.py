"""An agent's stored spawn descriptor is its own copy, built once per spawn.

The descriptor keeps the controller's construction config as plain data, a
fresh dict from ``ControllerParams.as_config()`` (``asdict`` copies nested
values), so it needs no second deep copy. Parameter introspection for spawn
factories and controller configs is cached per callable, without keeping a
class alive and without failing on callables that cannot be cached.
"""

from __future__ import annotations

import functools
import gc
import inspect
import logging
import types
import weakref
from typing import Any, Dict, cast

import pybullet as p
import pytest

from pybullet_fleet import Agent, AgentSpawnParams, MotionMode, MultiRobotSimulationCore, Pose, SimulationParams
from pybullet_fleet import config_utils
from pybullet_fleet import controller as controller_module
from pybullet_fleet.controller import OmniController


@pytest.fixture
def sim():
    sc = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, physics=False, monitor_log_interval=0))
    sc.initialize_simulation()
    yield sc
    p.disconnect(sc.client)


CONTROLLER = {"type": "omni", "max_linear_vel": [1.0, 0.5, 0.0], "max_linear_accel": 2.0}


def _spawn(sim, name):
    params = AgentSpawnParams(
        name=name,
        urdf_path="robots/simple_cube.urdf",
        initial_pose=Pose.from_xyz(0.0, 0.0, 0.1),
        motion_mode=MotionMode.OMNIDIRECTIONAL,
        controller=dict(CONTROLLER, max_linear_vel=list(CONTROLLER["max_linear_vel"])),
    )
    return Agent.from_params(params, sim_core=sim)


def _descriptor_controller(agent) -> Dict[str, Any]:
    """The controller config an agent's spawn descriptor keeps (a plain dict)."""
    return cast(Dict[str, Any], cast(AgentSpawnParams, agent._checkpoint_spawn_params).controller)


def test_a_stored_descriptor_is_independent(sim):
    """Mutating one agent's stored controller config changes neither the other
    agent's nor the live controller's limits."""
    first = _spawn(sim, "first")
    second = _spawn(sim, "second")
    stored = _descriptor_controller(first)
    assert stored["max_linear_vel"] == [1.0, 0.5, 0.0]

    stored["max_linear_vel"][0] = 99.0
    stored["max_linear_accel"] = 99.0

    assert _descriptor_controller(second)["max_linear_vel"] == [1.0, 0.5, 0.0]
    assert _descriptor_controller(second)["max_linear_accel"] == 2.0
    assert list(first.controller.params.max_linear_vel) == [1.0, 0.5, 0.0]
    assert first.controller.params.max_linear_accel == 2.0


@pytest.fixture
def signatures(monkeypatch):
    """The callables inspect.signature is asked about, with a cold cache."""
    calls = []
    real = config_utils.inspect.signature

    def counting(function):
        calls.append(function)
        return real(function)

    monkeypatch.setattr(config_utils, "_PARAMETER_CACHE", weakref.WeakKeyDictionary())
    monkeypatch.setattr(config_utils.inspect, "signature", counting)
    return calls


def test_parameter_names_are_introspected_once_per_callable(signatures):
    def factory(cls, name, pose=None, *, mass=0.0):  # a stand-in factory
        return None

    for _ in range(5):
        assert config_utils._parameter_names(factory) == frozenset({"name", "pose", "mass"})
    assert signatures == [factory]


def test_a_bound_classmethod_is_cached_under_its_function(signatures):
    """``Factory.make`` is a new bound method on every access; it still costs
    one introspection, and the bound first parameter is left out whatever its name."""

    class Factory:
        @classmethod
        def make(klass, name, pose=None):  # pyright: ignore[reportSelfClsParameterName]
            return None

    for _ in range(5):
        assert config_utils._parameter_names(Factory.make) == frozenset({"name", "pose"})
    assert signatures == [Factory.make.__func__]


def test_the_cache_does_not_keep_a_class_alive(monkeypatch):
    monkeypatch.setattr(config_utils, "_PARAMETER_CACHE", weakref.WeakKeyDictionary())

    def make_class():
        class Throwaway:
            def __init__(self, params, extra=None):
                pass

        return Throwaway

    cls = make_class()
    assert config_utils._parameter_names(cls.__init__) == frozenset({"params", "extra"})
    gone = weakref.ref(cls)
    del cls
    gc.collect()
    assert gone() is None
    assert len(config_utils._PARAMETER_CACHE) == 0


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _Unhashable(), id="unhashable"),
        pytest.param(lambda: functools.partial(_two, 1), id="partial"),
        pytest.param(lambda: [].append, id="builtin-bound"),
    ],
)
def test_any_callable_still_works(make):
    """Callables that cannot be weak dictionary keys are introspected uncached."""
    function = make()
    expected = frozenset(inspect.signature(function).parameters) - {"cls", "self"}
    for _ in range(2):
        assert config_utils._parameter_names(function) == expected


class _Unhashable:
    __hash__ = None  # type: ignore[assignment]

    def __call__(self, name, pose=None):
        return None


def _two(a, b, *, c=0):
    return None


class _HoldingController(OmniController):
    """Breaks the _construction_config contract: hands back a dict it keeps."""

    def __init__(self, params=None):
        super().__init__(params)
        self.held = {"type": "omni", "max_linear_vel": [1.0, 0.5, 0.0], "max_linear_accel": 2.0}

    def _construction_config(self):
        return self.held


def test_an_override_returning_a_held_dict_is_still_copied():
    controller = _HoldingController()
    stored = controller._spawn_descriptor_config()
    assert stored is not None and stored == controller.held
    stored["max_linear_vel"][0] = 99.0
    assert controller.held["max_linear_vel"][0] == 1.0


def test_the_built_in_config_is_not_copied_twice(monkeypatch):
    """The base implementation's fresh dict is kept as it is: no second copy."""
    controller = OmniController()
    fresh = controller._construction_config()
    monkeypatch.setattr(controller, "_construction_config", lambda: fresh)
    assert controller._spawn_descriptor_config() is fresh


class _DeclaredFreshController(OmniController):
    """Keeps the contract and declares it: a new dict on every call."""

    _construction_config_is_fresh = True
    calls = 0

    def _construction_config(self):
        type(self).calls += 1
        return {"type": "omni", "max_linear_vel": [1.0, 0.5, 0.0], "max_linear_accel": 2.0}


class _DeclaredButHoldingController(_HoldingController):
    """Declares the contract and breaks it: hands back a dict it keeps."""

    _construction_config_is_fresh = True


def test_a_declared_fresh_override_is_not_copied(monkeypatch):
    def no_copy(_):
        raise AssertionError("a declared fresh config was deep-copied")

    monkeypatch.setattr(controller_module, "copy", types.SimpleNamespace(deepcopy=no_copy))
    _DeclaredFreshController.calls = 0
    first = _DeclaredFreshController()._spawn_descriptor_config()
    second = _DeclaredFreshController()._spawn_descriptor_config()
    assert first == second and first is not second
    # Two calls on the first spawn of the class to check it, one afterwards.
    assert _DeclaredFreshController.calls == 3


def test_a_declared_override_that_holds_its_dict_warns_once_and_is_copied(caplog):
    with caplog.at_level(logging.WARNING, logger="pybullet_fleet"):
        controllers = [_DeclaredButHoldingController() for _ in range(3)]
        stored = [c._spawn_descriptor_config() for c in controllers]
    for controller, config in zip(controllers, stored):
        assert config is not None and config == controller.held and config is not controller.held
        config["max_linear_vel"][0] = 99.0
        assert controller.held["max_linear_vel"][0] == 1.0
    assert caplog.text.count("declares _construction_config_is_fresh") == 1


def test_an_undeclared_override_is_copied_without_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="pybullet_fleet"):
        controller = _HoldingController()
        assert controller._spawn_descriptor_config() is not controller.held
    assert "declares _construction_config_is_fresh" not in caplog.text
