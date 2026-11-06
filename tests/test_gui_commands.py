"""Tests for monitor-originated GUI commands without a GUI backend."""

import logging

import pybullet as p
import pytest
import numpy as np

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.data_monitor import DataMonitor
from pybullet_fleet.gui_commands import GuiCommand, GuiCommandType, MonitorFrame


def test_monitor_commands_are_consumed_at_step_boundary():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        sim.initialize_simulation()
        assert sim.enqueue_gui_command(GuiCommand(GuiCommandType.PAUSE))
        sim.step_once()
        assert sim.is_paused
        assert sim.step_count == 0

        assert sim.enqueue_gui_command(GuiCommand(GuiCommandType.SINGLE_STEP))
        sim.step_once()
        assert sim.is_paused
        assert sim.step_count == 1

        assert sim.enqueue_gui_command(GuiCommand(GuiCommandType.RESUME))
        sim.step_once()
        assert not sim.is_paused
        assert sim.step_count == 2
    finally:
        p.disconnect(sim.client)


def test_single_step_request_is_ignored_while_running():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        sim.initialize_simulation()
        assert not sim.request_single_step()
        sim.pause()
        assert sim.request_single_step(source="test")
        sim.step_once()
        assert sim.is_paused
        assert sim.step_count == 1
    finally:
        p.disconnect(sim.client)


def test_monitor_frame_is_read_only_summary():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        sim.initialize_simulation()
        sim.update_monitor()
        frame = sim.last_monitor_frame
        assert isinstance(frame, MonitorFrame)
        assert frame.steps == 0
        assert not frame.paused
        assert frame.agents == 0
    finally:
        p.disconnect(sim.client)


def test_data_monitor_submits_typed_commands_without_core_access():
    received = []
    monitor = DataMonitor(enable_gui=False)
    monitor.set_command_sink(lambda command: received.append(command) or True)

    monitor._submit_command(GuiCommandType.PAUSE)

    assert received == [GuiCommand(command=GuiCommandType.PAUSE)]


def test_data_monitor_reports_rejected_command_without_core_access():
    monitor = DataMonitor(enable_gui=False)
    monitor.set_command_sink(lambda command: False)

    assert not monitor._submit_command(GuiCommandType.PAUSE)


def test_data_monitor_can_clear_selection_without_core_access():
    received = []
    monitor = DataMonitor(enable_gui=False)
    monitor.set_command_sink(lambda command: received.append(command) or True)

    monitor._clear_selection()

    assert received == [GuiCommand(command=GuiCommandType.SELECT_ENTITY, entity_id=None)]


def test_monitor_selection_and_follow_are_applied_at_step_boundary():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        sim.initialize_simulation()
        from pybullet_fleet.sim_object import SimObject

        obj = SimObject.from_mesh(sim_core=sim, name="selected")
        sim.enqueue_gui_command(GuiCommand(GuiCommandType.SELECT_ENTITY, entity_id=obj.object_id))
        sim.enqueue_gui_command(GuiCommand(GuiCommandType.SET_FOLLOW, enabled=True))
        sim.step_once()
        assert sim.selected_entity_id == obj.object_id
        assert sim.gui_follow_enabled
        sim.update_monitor()
        assert sim.last_monitor_frame.selected_entity.entity.object_id == obj.object_id
    finally:
        p.disconnect(sim.client)


def test_pacing_change_is_applied_at_boundary_without_sim_time_jump():
    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, monitor=False, enable_floor=False, physics=False, timestep=0.1, target_rtf=1.0)
    )
    try:
        sim.initialize_simulation()
        sim.step_once()
        assert sim.step_count == 1
        sim.enqueue_gui_command(GuiCommand(GuiCommandType.SET_SIMULATION_PACING, target_rtf=2.0, timestep=0.2))
        sim.step_once()

        assert not sim.is_paused
        assert sim.params.target_rtf == 2.0
        assert sim.params.timestep == 0.2
        sim.update_monitor()
        assert sim.last_monitor_frame.sim_time == pytest.approx(0.3)
    finally:
        p.disconnect(sim.client)


def test_invalid_pacing_change_reports_every_invalid_value(caplog):
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        with caplog.at_level(logging.WARNING):
            sim._set_simulation_pacing(target_rtf=-1.0, timestep=0.0, source="test")

        assert "target_rtf=-1.0" in caplog.text
        assert "timestep=0.0" in caplog.text
    finally:
        p.disconnect(sim.client)


def test_paused_pose_edit_is_applied_at_step_boundary():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, physics=False))
    try:
        from pybullet_fleet.sim_object import SimObject

        sim.initialize_simulation()
        obj = SimObject.from_mesh(sim_core=sim, name="movable")
        sim.pause()
        sim.enqueue_gui_command(
            GuiCommand(
                GuiCommandType.SET_ENTITY_POSE,
                entity_id=obj.object_id,
                position=(2.0, 3.0, 0.4),
                rpy_radians=(0.2, -0.3, 1.0),
            )
        )
        sim.step_once()

        pose = obj.get_pose()
        assert list(pose.position) == [2.0, 3.0, 0.4]
        assert pose.roll == pytest.approx(0.2)
        assert pose.pitch == pytest.approx(-0.3)
        assert pose.yaw == pytest.approx(1.0)
        assert sim.is_paused
    finally:
        p.disconnect(sim.client)


def test_running_pose_edit_is_rejected():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, physics=False))
    try:
        from pybullet_fleet.sim_object import SimObject

        sim.initialize_simulation()
        obj = SimObject.from_mesh(sim_core=sim, name="movable")
        initial_position = list(obj.get_pose().position)
        sim.enqueue_gui_command(GuiCommand(GuiCommandType.SET_ENTITY_POSE, entity_id=obj.object_id, position=(2.0, 3.0, 0.4)))
        sim.step_once()

        assert list(obj.get_pose().position) == initial_position
    finally:
        p.disconnect(sim.client)


def test_viewport_drag_moves_the_selected_paused_entity(monkeypatch):
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, physics=False))
    try:
        from pybullet_fleet.geometry import Pose
        from pybullet_fleet.sim_object import SimObject

        sim.initialize_simulation()
        obj = SimObject.from_mesh(sim_core=sim, name="movable")
        obj.set_pose(Pose.from_xyz(1.0, 1.0, 0.4))
        sim.pause()
        sim._set_selected_entity(obj.object_id, source="test")
        monkeypatch.setattr(
            sim,
            "_viewport_ray_at",
            lambda mouse_x, mouse_y: (np.array([0.0, 0.0, 10.0]), np.array([10.0, 10.0, -10.0])),
        )

        sim._drag_selected_entity_to(50, 50)

        assert obj.get_pose().position == pytest.approx([4.8, 4.8, 0.4])
    finally:
        p.disconnect(sim.client)


def test_selected_entity_uses_one_world_coordinate_nameplate(monkeypatch):
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        from pybullet_fleet.geometry import Pose
        from pybullet_fleet.sim_object import SimObject

        sim.initialize_simulation()
        obj = SimObject.from_mesh(sim_core=sim, name="selected")
        obj.set_pose(Pose.from_xyz(2.0, 3.0, 0.2))
        added = []
        removed = []
        monkeypatch.setattr(sim._params, "gui", True)
        monkeypatch.setattr(
            p,
            "addUserDebugText",
            lambda text, position, **kwargs: added.append((text, position, kwargs)) or 42,
        )
        monkeypatch.setattr(p, "removeUserDebugItem", lambda item_id, **kwargs: removed.append((item_id, kwargs)))

        sim._set_selected_entity(obj.object_id, source="test")

        assert added == [
            (
                "selected",
                [2.0, 3.0, 0.5],
                {
                    "textColorRGB": [1, 1, 0],
                    "textSize": 1.2,
                    "replaceItemUniqueId": -1,
                    "physicsClientId": sim.client,
                },
            )
        ]
        obj.set_pose(Pose.from_xyz(4.0, 5.0, 0.4))
        sim._update_selected_nameplate(force=True)
        assert added[-1] == (
            "selected",
            [4.0, 5.0, 0.7],
            {
                "textColorRGB": [1, 1, 0],
                "textSize": 1.2,
                "replaceItemUniqueId": 42,
                "physicsClientId": sim.client,
            },
        )
        sim._set_selected_entity(None, source="test")
        assert removed == [(42, {"physicsClientId": sim.client})]
    finally:
        p.disconnect(sim.client)


def test_viewport_click_selects_without_an_existing_selection(monkeypatch):
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        from pybullet_fleet.sim_object import SimObject

        sim.initialize_simulation()
        obj = SimObject.from_mesh(sim_core=sim, name="selected")
        monkeypatch.setattr(sim, "_pick_entity_at", lambda mouse_x, mouse_y: obj.object_id)

        sim._select_viewport_entity(
            [
                (2, 100, 100, 0, 3),
                (2, 102, 104, 0, p.KEY_WAS_RELEASED),
            ]
        )

        assert sim.selected_entity_id == obj.object_id
    finally:
        p.disconnect(sim.client)


def test_viewport_click_on_empty_space_clears_selection(monkeypatch):
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    try:
        from pybullet_fleet.sim_object import SimObject

        sim.initialize_simulation()
        obj = SimObject.from_mesh(sim_core=sim, name="selected")
        sim._set_selected_entity(obj.object_id, source="test")
        monkeypatch.setattr(sim, "_pick_entity_at", lambda mouse_x, mouse_y: None)

        sim._select_viewport_entity([(2, 100, 100, 0, p.KEY_WAS_RELEASED)])

        assert sim.selected_entity_id is None
    finally:
        p.disconnect(sim.client)
