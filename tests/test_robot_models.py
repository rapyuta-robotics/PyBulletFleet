"""Tests for robot model resolution and auto-detection."""

import os

import pybullet as p
import pytest

from pybullet_fleet.robot_models import (
    KNOWN_MODELS,
    auto_detect_profile,
    list_all_models,
    resolve_model,
)

# Collect model names by tier from registry (automatically covers new entries)
_TIER1_MODELS = [name for name, entry in KNOWN_MODELS.items() if entry.tier == "pybullet_data"]
_ROS_MODELS = [name for name, entry in KNOWN_MODELS.items() if entry.tier == "ros"]
_RD_MODELS = [name for name, entry in KNOWN_MODELS.items() if entry.tier == "robot_descriptions"]


class TestResolveModel:
    """resolve_model() resolves robot names to absolute URDF/SDF paths."""

    @pytest.mark.parametrize("model_name", _TIER1_MODELS)
    def test_tier1_model_resolves_to_existing_file(self, model_name):
        path = resolve_model(model_name)
        assert os.path.isfile(path), f"{model_name} resolved to {path} but file does not exist"
        assert path.endswith((".urdf", ".sdf"))

    @pytest.mark.parametrize("model_name", _ROS_MODELS)
    def test_ros_model_raises_with_install_hint(self, model_name):
        """Tier 2 models raise FileNotFoundError with apt install hint (no ROS env in tests)."""
        with pytest.raises(FileNotFoundError, match="apt install"):
            resolve_model(model_name)

    @pytest.mark.parametrize("model_name", _RD_MODELS)
    def test_robot_descriptions_model_raises_with_install_hint(self, model_name):
        """Tier 3 models raise FileNotFoundError with pip install hint when package is missing."""
        try:
            import robot_descriptions  # noqa: F401

            pytest.skip("robot_descriptions is installed — cannot test missing-package error")
        except ImportError:
            pass
        with pytest.raises(FileNotFoundError, match="pip install"):
            resolve_model(model_name)

    def test_absolute_path_returned_as_is(self):
        path = resolve_model("/tmp/some_robot.urdf")
        assert path == "/tmp/some_robot.urdf"

    def test_bundled_relative_path_resolves_to_package(self, tmp_path, monkeypatch):
        # A bundled relative path resolves to the packaged asset, so it works for
        # pip-installed users whose CWD is not the repo (not just returned as-is).
        # chdir to an empty dir so a CWD-relative robots/ tree can't shadow it.
        monkeypatch.chdir(tmp_path)
        path = resolve_model("robots/arm_robot.urdf")
        assert os.path.isabs(path)
        assert path.endswith(os.path.join("robots", "arm_robot.urdf"))
        assert os.path.isfile(path)

    def test_unknown_relative_path_returned_as_is(self, tmp_path, monkeypatch):
        # A relative path not found in the CWD or the bundle is returned unchanged.
        # chdir to an empty dir so the relative path can't accidentally exist.
        monkeypatch.chdir(tmp_path)
        assert resolve_model("nowhere/custom_robot.urdf") == "nowhere/custom_robot.urdf"

    def test_unknown_name_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="nonexistent_robot"):
            resolve_model("nonexistent_robot")


class TestListAvailableModels:
    """list_all_models() reports model availability."""

    def test_returns_dict_with_all_known_models(self):
        result = list_all_models()
        assert isinstance(result, dict)
        assert set(result.keys()) == set(KNOWN_MODELS.keys())

    def test_tier1_models_all_available(self):
        result = list_all_models()
        for name in _TIER1_MODELS:
            assert result[name]["available"] is True, f"{name} should be available"
            assert result[name]["tier"] == "pybullet_data"

    def test_optional_model_resolution_error_is_reported(self, monkeypatch):
        """A failed optional asset download must not abort the whole catalog."""
        import pybullet_fleet.robot_models as robot_models

        def fail_for_one_model(name):
            if name == "pr2":
                raise OSError("optional asset cache is unavailable")
            return "/tmp/available.urdf"

        monkeypatch.setattr(robot_models, "resolve_model", fail_for_one_model)
        result = robot_models.list_all_models()

        assert result["pr2"] == {
            "tier": "robot_descriptions",
            "available": False,
            "error": "optional asset cache is unavailable",
        }


class TestAutoDetectProfile:
    """auto_detect_profile() introspects URDF via PyBullet.

    The parametrized sweep (test_tier1_model_profile_has_valid_fields) covers
    ALL Tier 1 models.  Below that, only ONE representative per robot_type
    category is tested for specific field values.
    """

    @pytest.fixture(autouse=True)
    def _setup_pybullet(self):
        self.client = p.connect(p.DIRECT)
        yield
        p.disconnect(self.client)

    # -- Parametrized sweep: covers every Tier 1 model -------------------

    @pytest.mark.parametrize("model_name", _TIER1_MODELS)
    def test_tier1_model_profile_has_valid_fields(self, model_name):
        """Every Tier 1 model loads and produces a valid profile."""
        path = resolve_model(model_name)
        profile = auto_detect_profile(path, self.client)
        assert profile.urdf_path == path
        assert profile.robot_type in ("arm", "mobile", "mobile_manipulator", "quadruped", "object", "unknown")
        assert profile.num_joints >= 0
        assert len(profile.joint_max_velocities) == len(profile.movable_joint_names)

    # -- One representative per robot_type category ----------------------

    def test_object_type_table(self):
        """Zero-joint model → 'object', num_joints == 0."""
        profile = auto_detect_profile(resolve_model("table"), self.client)
        assert profile.robot_type == "object"
        assert profile.num_joints == 0

    def test_quadruped_type_a1(self):
        """Quadruped → 'quadruped', no EE."""
        profile = auto_detect_profile(resolve_model("a1"), self.client)
        assert profile.robot_type == "quadruped"
        assert profile.ee_link_name is None

    def test_arm_type_panda(self):
        """Arm → 'arm', EE resolved to 'panda_hand', ee_link_index valid."""
        profile = auto_detect_profile(resolve_model("panda"), self.client)
        assert profile.robot_type == "arm"
        assert profile.ee_link_name == "panda_hand"
        assert profile.ee_link_index >= 0
        # Verify ee_link_index resolves to the correct link name in PyBullet
        body_id = p.loadURDF(resolve_model("panda"), useFixedBase=True, physicsClientId=self.client)
        link_name = p.getJointInfo(body_id, profile.ee_link_index, physicsClientId=self.client)[12].decode("utf-8")
        p.removeBody(body_id, physicsClientId=self.client)
        assert link_name == profile.ee_link_name

    def test_mobile_type_husky(self):
        """Mobile → 'mobile', no EE, ee_link_index == -1."""
        profile = auto_detect_profile(resolve_model("husky"), self.client)
        assert profile.robot_type == "mobile"
        assert profile.ee_link_name is None
        assert profile.ee_link_index == -1

    # -- Override mechanism ----------------------------------------------

    def test_override_merges_with_detected(self):
        profile = auto_detect_profile(resolve_model("panda"), self.client, ee_link_name="custom_link")
        assert profile.ee_link_name == "custom_link"
        assert profile.robot_type == "arm"  # still auto-detected

    def test_override_robot_type(self):
        profile = auto_detect_profile(resolve_model("panda"), self.client, robot_type="mobile_manipulator")
        assert profile.robot_type == "mobile_manipulator"

    # -- Profile fields: joint indices, targets, arm_reach ---------------

    def test_movable_joint_indices_match_names(self):
        """_movable_joint_indices has same length as movable_joint_names."""
        profile = auto_detect_profile(resolve_model("panda"), self.client)
        assert len(profile._movable_joint_indices) == len(profile.movable_joint_names)
        assert len(profile._movable_joint_indices) == 9  # 7 arm + 2 finger

    def test_make_joint_targets_panda(self):
        """make_joint_targets produces correct-length list from fractions."""
        from pybullet_fleet.agent import Agent

        path = resolve_model("panda")
        profile = auto_detect_profile(path, self.client)
        agent = Agent.from_urdf(urdf_path=path, use_fixed_base=True, sim_core=None)
        n_movable = len(profile.movable_joint_names)

        # Zero fractions → midpoints
        targets = profile.make_joint_targets([0.0] * n_movable, agent)
        assert len(targets) == len(agent.joint_info)  # 12 for panda

        # Non-zero fractions → within limits
        targets_pos = profile.make_joint_targets([0.5] * n_movable, agent)
        for idx, (lo, hi) in zip(profile._movable_joint_indices, zip(profile.joint_lower_limits, profile.joint_upper_limits)):
            assert lo <= targets_pos[idx] <= hi

    def test_arm_reach_computed_for_arm(self):
        """arm_reach is computed for arm-type robots and within expected range."""
        profile = auto_detect_profile(resolve_model("panda"), self.client)
        assert profile.arm_reach is not None
        assert 0.8 < profile.arm_reach < 1.5

    def test_arm_reach_none_for_mobile(self):
        """arm_reach is None for mobile-type robots."""
        profile = auto_detect_profile(resolve_model("husky"), self.client)
        assert profile.arm_reach is None

    # -- body_id overload: introspect without load/remove ----------------

    def test_body_id_returns_same_profile_as_path(self):
        """Passing body_id produces the same profile as passing urdf_path."""
        path = resolve_model("panda")
        profile_from_path = auto_detect_profile(path, self.client)

        body_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.client)
        profile_from_body = auto_detect_profile(body_id, self.client)
        p.removeBody(body_id, physicsClientId=self.client)

        assert profile_from_body.robot_type == profile_from_path.robot_type
        assert profile_from_body.ee_link_name == profile_from_path.ee_link_name
        assert profile_from_body.num_joints == profile_from_path.num_joints
        assert profile_from_body.movable_joint_names == profile_from_path.movable_joint_names

    def test_body_id_does_not_remove_body(self):
        """When body_id is passed, the body must still exist after the call."""
        path = resolve_model("panda")
        body_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.client)

        auto_detect_profile(body_id, self.client)

        # Body should still be alive — getBodyInfo would raise if removed
        info = p.getBodyInfo(body_id, physicsClientId=self.client)
        assert info is not None
        p.removeBody(body_id, physicsClientId=self.client)

    def test_body_id_restores_joint_positions(self):
        """Probing the arm reach must leave every movable joint's position and velocity as they were."""
        path = resolve_model("panda")
        body_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.client)
        movable = [
            j
            for j in range(p.getNumJoints(body_id, physicsClientId=self.client))
            if p.getJointInfo(body_id, j, physicsClientId=self.client)[2] != p.JOINT_FIXED
        ]
        assert movable
        before = {}
        for k, j in enumerate(movable):
            # Non-zero velocity too: the restore must put back (position, velocity).
            p.resetJointState(body_id, j, 0.1 * (k + 1), 0.05 * (k + 1), physicsClientId=self.client)
            before[j] = p.getJointState(body_id, j, physicsClientId=self.client)[:2]
            assert before[j][1] != 0.0

        auto_detect_profile(body_id, self.client)

        after = {j: p.getJointState(body_id, j, physicsClientId=self.client)[:2] for j in movable}
        p.removeBody(body_id, physicsClientId=self.client)
        for j in movable:
            assert after[j] == pytest.approx(before[j])

    def test_body_id_restores_joint_positions_when_probing_raises(self, monkeypatch):
        """A probe that fails partway still restores every movable joint's position and velocity."""
        import pybullet_fleet.robot_models as robot_models

        path = resolve_model("panda")
        body_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.client)
        movable = [
            j
            for j in range(p.getNumJoints(body_id, physicsClientId=self.client))
            if p.getJointInfo(body_id, j, physicsClientId=self.client)[2] != p.JOINT_FIXED
        ]
        assert movable
        before = {}
        for k, j in enumerate(movable):
            # Non-zero velocity too: the restore must put back (position, velocity).
            p.resetJointState(body_id, j, 0.1 * (k + 1), 0.05 * (k + 1), physicsClientId=self.client)
            before[j] = p.getJointState(body_id, j, physicsClientId=self.client)[:2]
            assert before[j][1] != 0.0

        real_get_link_state = robot_models.p.getLinkState
        calls = []

        def failing_get_link_state(*args, **kwargs):
            calls.append(args)
            if len(calls) >= 2:  # the first probe config has already moved the joints
                raise RuntimeError("probe failed")
            return real_get_link_state(*args, **kwargs)

        monkeypatch.setattr(robot_models.p, "getLinkState", failing_get_link_state)
        with pytest.raises(RuntimeError, match="probe failed"):
            auto_detect_profile(body_id, self.client)
        monkeypatch.undo()

        after = {j: p.getJointState(body_id, j, physicsClientId=self.client)[:2] for j in movable}
        p.removeBody(body_id, physicsClientId=self.client)
        for j in movable:
            assert after[j] == pytest.approx(before[j])

    def test_body_id_with_overrides(self):
        """Overrides work with body_id input too."""
        path = resolve_model("panda")
        body_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.client)

        profile = auto_detect_profile(body_id, self.client, robot_type="mobile")
        p.removeBody(body_id, physicsClientId=self.client)

        assert profile.robot_type == "mobile"


class TestResolveXacroCacheKey:
    """_resolve_xacro uses xacro_args in cache key so different params don't collide."""

    def test_different_xacro_args_produce_different_cache_files(self, tmp_path, monkeypatch):
        """Same xacro file with different args must not return a stale cached URDF."""
        import pybullet_fleet.robot_models as rm

        # Create a minimal xacro file (content doesn't matter — we mock processing)
        xacro_file = tmp_path / "test.urdf.xacro"
        xacro_file.write_text("<robot/>")

        # Redirect cache dir to tmp_path
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        # Track what xacro_args each call receives by writing them into the "URDF"
        call_args_log = []

        def fake_xacro_process(xacro_path, mappings=None):
            """Fake xacro.process_file that returns XML containing the mappings."""
            from unittest.mock import MagicMock

            call_args_log.append(mappings)
            doc = MagicMock()
            doc.toxml.return_value = f"<robot><!-- args={mappings} --></robot>"
            return doc

        # Patch xacro import to use our fake
        import types

        fake_xacro = types.ModuleType("xacro")
        fake_xacro.process_file = fake_xacro_process  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "xacro", fake_xacro)

        # Also stub _resolve_package_uris to no-op (no ROS on host)
        monkeypatch.setattr(rm, "_resolve_package_uris", lambda xml: xml)

        path_a = rm._resolve_xacro("test_robot", str(xacro_file), "flavor:=chocolate")
        path_b = rm._resolve_xacro("test_robot", str(xacro_file), "flavor:=strawberry")

        # The two calls must not return the same cached file
        assert path_a != path_b, "Different xacro_args returned the same cache path — args not in cache key"


class TestSearchPath:
    """add_search_path() / remove_search_path() / get_search_paths()."""

    def test_add_search_path_resolves_name(self, tmp_path):
        """A URDF in an added search path is resolved by stem name."""
        from pybullet_fleet.robot_models import add_search_path, remove_search_path

        # Create a URDF file in a temp directory
        urdf = tmp_path / "my_custom_agv.urdf"
        urdf.write_text("<robot name='my_custom_agv'/>")

        add_search_path(str(tmp_path))
        try:
            path = resolve_model("my_custom_agv")
            assert path == str(urdf)
        finally:
            remove_search_path(str(tmp_path))

    def test_search_path_takes_priority_over_known_models(self, tmp_path):
        """User search paths are checked before KNOWN_MODELS registry."""
        from pybullet_fleet.robot_models import add_search_path, remove_search_path

        # Create a file named "panda.urdf" in user dir — should shadow built-in
        urdf = tmp_path / "panda.urdf"
        urdf.write_text("<robot name='my_panda_override'/>")

        add_search_path(str(tmp_path))
        try:
            path = resolve_model("panda")
            assert path == str(urdf), "User search path should take priority over KNOWN_MODELS"
        finally:
            remove_search_path(str(tmp_path))

    def test_remove_search_path_restores_default(self, tmp_path):
        """After removing a search path, resolve falls back to KNOWN_MODELS."""
        from pybullet_fleet.robot_models import add_search_path, remove_search_path

        urdf = tmp_path / "panda.urdf"
        urdf.write_text("<robot name='my_panda_override'/>")

        add_search_path(str(tmp_path))
        remove_search_path(str(tmp_path))

        # Should resolve to pybullet_data panda, not the tmp one
        path = resolve_model("panda")
        assert str(tmp_path) not in path

    def test_get_search_paths_returns_current_list(self, tmp_path):
        """get_search_paths() returns the current search path list."""
        from pybullet_fleet.robot_models import add_search_path, get_search_paths, remove_search_path

        test_dir = str(tmp_path / "test_robots")
        os.makedirs(test_dir)

        original = list(get_search_paths())
        add_search_path(test_dir)
        try:
            paths = get_search_paths()
            assert test_dir in paths
        finally:
            remove_search_path(test_dir)
        assert get_search_paths() == original

    def test_add_duplicate_is_idempotent(self, tmp_path):
        """Adding the same path twice does not duplicate it."""
        from pybullet_fleet.robot_models import add_search_path, get_search_paths, remove_search_path

        add_search_path(str(tmp_path))
        add_search_path(str(tmp_path))
        try:
            count = list(get_search_paths()).count(str(tmp_path))
            assert count == 1, "Duplicate search path should not be added"
        finally:
            remove_search_path(str(tmp_path))

    def test_sdf_file_also_found(self, tmp_path):
        """Search paths also find .sdf files by stem name."""
        from pybullet_fleet.robot_models import add_search_path, remove_search_path

        sdf = tmp_path / "warehouse.sdf"
        sdf.write_text("<sdf/>")

        add_search_path(str(tmp_path))
        try:
            path = resolve_model("warehouse")
            assert path == str(sdf)
        finally:
            remove_search_path(str(tmp_path))

    def test_nonexistent_dir_raises_value_error(self):
        """Adding a non-existent directory raises ValueError."""
        from pybullet_fleet.robot_models import add_search_path

        with pytest.raises(ValueError, match="does not exist"):
            add_search_path("/nonexistent/path/to/robots")


class TestRegisterModel:
    """register_model() / unregister_model()."""

    def test_register_model_resolves_by_name(self, tmp_path):
        """A registered model can be resolved by name."""
        from pybullet_fleet.robot_models import register_model, unregister_model

        urdf = tmp_path / "custom_bot.urdf"
        urdf.write_text("<robot name='custom_bot'/>")

        register_model("custom_bot", str(urdf))
        try:
            path = resolve_model("custom_bot")
            assert path == str(urdf)
        finally:
            unregister_model("custom_bot")

    def test_register_model_with_model_entry(self, tmp_path):
        """register_model() accepts a ModelEntry for tier metadata."""
        from pybullet_fleet.robot_models import ModelEntry, register_model, unregister_model

        register_model("my_arm", ModelEntry("my_arm/model.urdf", "pybullet_data"))
        try:
            from pybullet_fleet.robot_models import KNOWN_MODELS

            assert "my_arm" in KNOWN_MODELS
            assert KNOWN_MODELS["my_arm"].tier == "pybullet_data"
        finally:
            unregister_model("my_arm")

    def test_unregister_removes_model(self, tmp_path):
        """unregister_model() removes the model from registry."""
        from pybullet_fleet.robot_models import register_model, unregister_model

        urdf = tmp_path / "temp_bot.urdf"
        urdf.write_text("<robot name='temp_bot'/>")

        register_model("temp_bot", str(urdf))
        unregister_model("temp_bot")

        with pytest.raises(FileNotFoundError):
            resolve_model("temp_bot")

    def test_unregister_unknown_is_noop(self):
        """unregister_model() for unknown name does not raise."""
        from pybullet_fleet.robot_models import unregister_model

        # Should not raise
        unregister_model("nonexistent_model_xyz_999")

    def test_register_does_not_overwrite_builtin_by_default(self):
        """Registering a name that exists in KNOWN_MODELS raises ValueError."""
        from pybullet_fleet.robot_models import register_model

        with pytest.raises(ValueError, match="already registered"):
            register_model("panda", "/some/path.urdf")

    def test_register_overwrites_with_force(self, tmp_path):
        """register_model(force=True) overwrites existing entries."""
        from pybullet_fleet.robot_models import register_model, unregister_model

        urdf = tmp_path / "panda.urdf"
        urdf.write_text("<robot name='custom_panda'/>")

        original_path = resolve_model("panda")
        register_model("panda", str(urdf), force=True)
        try:
            path = resolve_model("panda")
            assert path == str(urdf)
        finally:
            # Restore original by unregistering custom (built-in won't be restored)
            unregister_model("panda")
            # Re-register built-in panda
            from pybullet_fleet.robot_models import KNOWN_MODELS, ModelEntry

            KNOWN_MODELS["panda"] = ModelEntry("franka_panda/panda.urdf", "pybullet_data")


class TestAutoDiscovery:
    """Fallback auto-discovery when model is not in KNOWN_MODELS."""

    def test_pybullet_data_unlisted_model_resolves(self):
        """A pybullet_data URDF not in KNOWN_MODELS is found by fallback scan."""
        # "r2d2" exists in pybullet_data but is NOT in KNOWN_MODELS
        assert "r2d2" not in KNOWN_MODELS
        path = resolve_model("r2d2")
        assert os.path.isfile(path)
        assert "r2d2" in path

    def test_pybullet_data_subfolder_model_resolves(self):
        """A pybullet_data model in a subdirectory is found by fallback scan."""
        # "humanoid" exists in pybullet_data/humanoid/humanoid.urdf
        assert "humanoid" not in KNOWN_MODELS
        path = resolve_model("humanoid")
        assert os.path.isfile(path)
        assert path.endswith(".urdf")

    def test_robot_descriptions_unlisted_model_resolves(self):
        """A robot_descriptions model not in KNOWN_MODELS is found by fallback scan."""
        pytest.importorskip("robot_descriptions")
        # "panda_description" is a well-known module; "panda" is in KNOWN_MODELS
        # but "ur10e" / "anymal_b" etc. are not — pick one that has URDF_PATH
        assert "anymal_b" not in KNOWN_MODELS
        path = resolve_model("anymal_b")
        assert os.path.isfile(path)
        assert path.endswith(".urdf")

    def test_truly_unknown_model_still_raises(self):
        """A name that doesn't exist anywhere still raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            resolve_model("completely_nonexistent_robot_xyz_999")

    def test_known_models_not_polluted_by_fallback(self):
        """Fallback discovery should NOT add entries to KNOWN_MODELS."""
        assert "r2d2" not in KNOWN_MODELS
        resolve_model("r2d2")
        # KNOWN_MODELS should remain unchanged
        assert "r2d2" not in KNOWN_MODELS

    def test_discover_pybullet_data_returns_dict(self):
        """discover_models('pybullet_data') returns a dict of found models."""
        from pybullet_fleet.robot_models import discover_models

        result = discover_models("pybullet_data")
        assert isinstance(result, dict)
        assert "r2d2" in result
        assert "humanoid" in result
        # Each value should be a path string
        for name, path in result.items():
            assert isinstance(path, str)

    def test_discover_robot_descriptions_returns_dict(self):
        """discover_models('robot_descriptions') returns available URDF models."""
        pytest.importorskip("robot_descriptions")
        from pybullet_fleet.robot_models import discover_models

        result = discover_models("robot_descriptions")
        assert isinstance(result, dict)
        assert len(result) > 0
        # Should include modules with URDF_PATH
        # anymal_b_description has a URDF
        assert "anymal_b" in result

    def test_discover_unknown_tier_raises(self):
        """discover_models() with unknown tier raises ValueError."""
        from pybullet_fleet.robot_models import discover_models

        with pytest.raises(ValueError, match="Unknown tier"):
            discover_models("nonexistent_tier")
