# tests/test_mesh_loader.py
"""Tests for load_mesh_directory() and backward-compat load_rmf_world alias.

Functions live in sdf_loader.py; world_loader.py re-exports for compatibility.
"""

import pybullet as p
import pytest

from pybullet_fleet.sim_object import SimObject
from pybullet_fleet.types import CollisionMode
from pybullet_fleet.sdf_loader import load_mesh_directory, load_rmf_world


class TestLoadMeshDirectory:
    """Tests for load_mesh_directory() OBJ directory loader."""

    def test_loads_obj_files(self, sim_core, tmp_path):
        """Loads OBJ files from directory."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "wall.obj").write_text(obj_content)
        (tmp_path / "floor.obj").write_text(obj_content)

        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core)
        assert len(objects) == 2
        assert all(isinstance(o, SimObject) for o in objects)

    def test_objects_are_static(self, sim_core, tmp_path):
        """All loaded objects are static (mass=0)."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "wall.obj").write_text(obj_content)

        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core)
        for obj in objects:
            assert obj.mass == 0.0

    def test_returns_list(self, sim_core, tmp_path):
        """Return type is always List[SimObject]."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "mesh.obj").write_text(obj_content)

        result = load_mesh_directory(str(tmp_path), sim_core=sim_core)
        assert isinstance(result, list)

    def test_empty_dir_returns_empty_list(self, sim_core, tmp_path):
        """Empty directory returns empty list."""
        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core)
        assert objects == []

    def test_nonexistent_dir_raises(self, sim_core):
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_mesh_directory("/nonexistent/path", sim_core=sim_core)

    def test_names_from_filenames(self, sim_core, tmp_path):
        """Object names come from OBJ filenames."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "L1_walls.obj").write_text(obj_content)

        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core)
        assert objects[0].name == "L1_walls"

    def test_objects_registered_in_sim_core(self, sim_core, tmp_path):
        """Loaded objects are registered in sim_core."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "wall.obj").write_text(obj_content)

        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core)
        for obj in objects:
            assert obj in sim_core.sim_objects

    def test_collision_mode_parameter(self, sim_core, tmp_path):
        """collision_mode is applied to loaded objects."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "wall.obj").write_text(obj_content)

        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core, collision_mode=CollisionMode.DISABLED)
        for obj in objects:
            assert obj.collision_mode == CollisionMode.DISABLED

    def test_custom_pattern(self, sim_core, tmp_path):
        """Custom glob pattern works."""
        (tmp_path / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        (tmp_path / "mesh.stl").write_bytes(b"")  # Ignored

        objects = load_mesh_directory(str(tmp_path), sim_core=sim_core, pattern="*.obj")
        assert len(objects) == 1

    def test_without_sim_core(self, tmp_path):
        """Works without sim_core (standalone)."""
        cid = p.connect(p.DIRECT)
        try:
            obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            (tmp_path / "wall.obj").write_text(obj_content)
            objects = load_mesh_directory(str(tmp_path))
            assert len(objects) == 1
        finally:
            p.disconnect(cid)

    def test_load_rmf_world_alias(self, sim_core, tmp_path):
        """load_rmf_world backward-compat alias delegates to load_mesh_directory."""
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        (tmp_path / "wall.obj").write_text(obj_content)

        objects = load_rmf_world(str(tmp_path), sim_core=sim_core)
        assert len(objects) == 1
