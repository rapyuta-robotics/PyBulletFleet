"""The scripted scenes of the collision parity trace.

Each scene is small (a few bodies, a few dozen steps) and deterministic, and
exercises one part of what the collision pipeline promises: the broad phase
across cells, the per-axis clearance and margin changes, collision mode
changes, static bodies, attachment, the ignore filter, physics bodies under
``CONTACT_POINTS`` and ``HYBRID``, an arm touching with its links, agents on paths, and a
collision check that runs every other step. Every object is named: the trace refers to
objects by name.
"""

from __future__ import annotations

from typing import Dict, Iterator, Sequence, Tuple

from pybullet_fleet.action import MoveAction
from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.core_simulation import MultiRobotSimulationCore
from pybullet_fleet.geometry import Path, Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode, SpatialHashCellSizeMode

from .trace import Scene

SCENES: Dict[str, Scene] = {}

KINEMATIC = dict(
    physics=False,
    timestep=0.1,
    collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
    collision_margin=0.05,
    spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
    spatial_hash_cell_size=1.0,
    ignore_static_collision=False,
)

PHYSICS = dict(
    physics=True,
    timestep=1.0 / 60.0,
    collision_detection_method=CollisionDetectionMethod.CONTACT_POINTS,
    collision_margin=0.02,
    spatial_hash_cell_size_mode=SpatialHashCellSizeMode.CONSTANT,
    spatial_hash_cell_size=2.0,
    ignore_static_collision=False,
)


def scene(name: str, **params):
    """Register the decorated generator as the scene ``name`` with these simulation parameters."""

    def register(script):
        SCENES[name] = Scene(name=name, params=params, script=script, doc=(script.__doc__ or "").strip())
        return script

    return register


# --- helpers ----------------------------------------------------------------


def box(
    sim: MultiRobotSimulationCore,
    name: str,
    xyz: Sequence[float],
    half: Sequence[float] = (0.2, 0.2, 0.2),
    mode: CollisionMode = CollisionMode.NORMAL_3D,
    mass: float = 0.0,
    rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> SimObject:
    shape = ShapeParams(shape_type="box", half_extents=list(half))
    return SimObject.from_mesh(
        visual_shape=shape,
        collision_shape=shape,
        pose=Pose.from_euler(*xyz, *rpy),
        mass=mass,
        sim_core=sim,
        collision_mode=mode,
        name=name,
    )


def robot(sim: MultiRobotSimulationCore, name: str, xy: Tuple[float, float], yaw: float = 0.0) -> Agent:
    return Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/mobile_robot.urdf",
            initial_pose=Pose.from_euler(xy[0], xy[1], 0.0, 0.0, 0.0, yaw),
            collision_mode=CollisionMode.NORMAL_3D,
            name=name,
        ),
        sim_core=sim,
    )


def shift(obj: SimObject, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
    """Move ``obj`` by a world offset, keeping its orientation."""
    pose = obj.get_pose()
    x, y, z = pose.position
    obj.set_pose(Pose(position=[x + dx, y + dy, z + dz], orientation=list(pose.orientation)))


def go_to(agent: Agent, x: float, y: float) -> None:
    agent.add_action(MoveAction(path=Path.from_positions([[x, y, 0.0]])))


# --- scenes -----------------------------------------------------------------


@scene("crossing_lanes", **KINEMATIC)
def crossing_lanes(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """Two lanes of boxes pass each other within the margin; one box drives into
    a static wall that spans several cells; one box far away never moves."""
    box(sim, "wall", (0.0, 2.0, 0.5), half=(3.0, 0.1, 0.5), mode=CollisionMode.STATIC)
    box(sim, "far", (8.0, 8.0, 0.2))
    east = [box(sim, f"east{i}", (-2.0 + i, 0.0, 0.2)) for i in range(3)]
    west = [box(sim, f"west{i}", (2.0 - i, 0.44, 0.2)) for i in range(3)]
    north = box(sim, "north", (-3.0, 0.8, 0.2))
    for step in range(40):
        yield
        for obj in east:
            shift(obj, dx=0.15)
        for obj in west:
            shift(obj, dx=-0.15)
        if step < 30:
            shift(north, dy=0.05)


@scene("per_axis_clearance", **dict(KINEMATIC, collision_margin=(0.05, 0.05, 0.0)))
def per_axis_clearance(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """A cart passes 2 cm under a shelf: reported only while the z clearance is
    non-zero. Boxes 4 cm apart side by side are reported until the clearance
    shrinks below that; boxes 8 cm apart only while it is 10 cm. The margin
    changes through the setter and through ``params`` while nothing moves, so
    stationary pairs are judged again."""
    box(sim, "shelf", (0.0, 0.0, 0.47), half=(0.4, 0.4, 0.05), mode=CollisionMode.STATIC)
    cart = box(sim, "cart", (-1.5, 0.0, 0.2))
    box(sim, "left", (0.0, 2.0, 0.2))
    box(sim, "right", (0.44, 2.0, 0.2))
    box(sim, "tall", (0.0, 2.0, 0.62))
    box(sim, "near", (0.0, 3.0, 0.2))
    box(sim, "nearer", (0.48, 3.0, 0.2))
    for step in range(45):
        yield
        if step < 30:
            shift(cart, dx=0.1)
        if step == 10:
            sim.set_collision_margin(0.05)
        if step == 20:
            sim.set_collision_margin((0.05, 0.05, 0.0))
        if step == 32:
            sim.params.collision_margin = (0.1, 0.1, 0.0)
        if step == 38:
            sim.params.collision_margin = 0.01


@scene("collision_modes", **KINEMATIC)
def collision_modes(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """A touching chain whose middle body is disabled and enabled again, whose
    end becomes static and normal again, and a body disabled while touching,
    enabled again and driven away."""
    a = box(sim, "a", (0.0, 0.0, 0.2))
    b = box(sim, "b", (0.38, 0.0, 0.2))
    box(sim, "c", (0.76, 0.0, 0.2))
    d = box(sim, "d", (0.0, 1.0, 0.2))
    for step in range(36):
        yield
        if step == 4:
            b.set_collision_mode(CollisionMode.DISABLED)
        if step == 9:
            b.set_collision_mode(CollisionMode.NORMAL_3D)
        if step == 14:
            a.set_collision_mode(CollisionMode.STATIC)
        if step == 19:
            a.set_collision_mode(CollisionMode.NORMAL_3D)
        if 10 <= step < 25:
            shift(d, dy=-0.05)
        if step == 24:
            d.set_collision_mode(CollisionMode.DISABLED)
        if step == 27:
            d.set_collision_mode(CollisionMode.NORMAL_3D)
        if 28 <= step < 34:
            shift(d, dy=0.1)


@scene("static_ignored", **dict(KINEMATIC, ignore_static_collision=True))
def static_ignored(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """With static collisions ignored, a pair with a static body is not
    reported until that body becomes normal; two touching static bodies never are."""
    a = box(sim, "a", (0.0, 0.0, 0.2), mode=CollisionMode.STATIC)
    box(sim, "b", (0.38, 0.0, 0.2))
    box(sim, "s1", (0.0, 2.0, 0.2), mode=CollisionMode.STATIC)
    box(sim, "s2", (0.38, 2.0, 0.2), mode=CollisionMode.STATIC)
    mover = box(sim, "mover", (-1.0, 0.0, 0.2))
    for step in range(20):
        yield
        if step == 4:
            a.set_collision_mode(CollisionMode.NORMAL_3D)
        if step == 12:
            a.set_collision_mode(CollisionMode.STATIC)
        if 6 <= step < 16:
            shift(mover, dx=0.08)


@scene("attach_detach", **KINEMATIC)
def attach_detach(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """A carrier with a load resting on it, and a second load on the first: the
    pairs are reported until attached, carried past a post within the margin
    (the post is reported against every member of the tree), then detached."""
    carrier = box(sim, "carrier", (0.0, 0.0, 0.2))
    load = box(sim, "load", (0.0, 0.0, 0.6))
    load2 = box(sim, "load2", (0.0, 0.0, 1.0))
    box(sim, "post", (1.5, 0.32, 0.6), half=(0.1, 0.1, 0.6), mode=CollisionMode.STATIC)
    for step in range(40):
        yield
        if step == 3:
            carrier.attach_object(load, keep_world_pose=True)
        if step == 5:
            load.attach_object(load2, keep_world_pose=True)
        if 6 <= step < 26:
            shift(carrier, dx=0.1)
        if step == 28:
            carrier.detach_object(load)
        if 29 <= step < 36:
            shift(carrier, dx=0.1)
        if step == 36:
            load.detach_object(load2)


@scene("ignore_filter", **KINEMATIC)
def ignore_filter(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """An ignore filter set, removed and replaced while the bodies stand still,
    and a body moved into contact while its pair is ignored."""
    box(sim, "a", (0.0, 0.0, 0.2))
    box(sim, "b", (0.38, 0.0, 0.2))
    box(sim, "c", (0.0, 2.0, 0.2))
    box(sim, "d", (0.38, 2.0, 0.2))
    e = box(sim, "e", (0.0, 4.0, 0.2))
    box(sim, "f", (0.8, 4.0, 0.2))

    def names(x, y):
        return {x.name, y.name}

    for step in range(30):
        yield
        if step == 4:
            sim.set_collision_ignore_filter(lambda x, y: names(x, y) == {"a", "b"})
        if step == 12:
            sim.set_collision_ignore_filter(None)
        if step == 18:
            sim.set_collision_ignore_filter(lambda x, y: names(x, y) in ({"c", "d"}, {"e", "f"}))
        if 19 <= step < 27:
            shift(e, dx=0.05)
        if step == 27:
            sim.set_collision_ignore_filter(None)


@scene("check_every_other_step", **dict(KINEMATIC, collision_check_frequency=5.0))
def check_every_other_step(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """The collision check runs every second step, so the moves of two steps
    are judged together and the observation keeps the earlier step."""
    a = box(sim, "a", (-1.0, 0.0, 0.2))
    box(sim, "b", (0.0, 0.0, 0.2))
    c = box(sim, "c", (0.0, 0.44, 0.2))
    for step in range(21):
        yield
        shift(a, dx=0.1)
        if step % 3 == 0:
            shift(c, dx=0.2)


@scene("physics_contact", **PHYSICS)
def physics_contact(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """Two dynamic boxes fall onto a static floor, one onto the other, and a
    kinematic box is pushed into the pile and pulled back."""
    box(sim, "floor", (0.0, 0.0, -0.1), half=(2.0, 2.0, 0.1), mode=CollisionMode.STATIC)
    box(sim, "drop1", (0.0, 0.0, 0.5), mass=1.0)
    box(sim, "drop2", (0.05, 0.0, 1.2), mass=1.0)
    pusher = box(sim, "pusher", (1.5, 0.0, 0.2))
    for step in range(60):
        yield
        if 30 <= step < 50:
            shift(pusher, dx=-0.06)
        if 54 <= step < 58:
            shift(pusher, dx=0.1)


@scene("hybrid_mixed", **dict(PHYSICS, collision_detection_method=CollisionDetectionMethod.HYBRID, collision_margin=0.05))
def hybrid_mixed(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """Under HYBRID, two kinematic boxes 4 cm apart are a near miss, while a
    kinematic box approaching a dynamic one is reported only on contact, and
    released when it backs off."""
    box(sim, "floor", (0.0, 0.0, -0.1), half=(2.0, 2.0, 0.1), mode=CollisionMode.STATIC)
    box(sim, "dyn", (0.0, 0.0, 0.5), mass=1.0)
    box(sim, "k1", (0.0, 1.5, 0.3))
    box(sim, "k2", (0.44, 1.5, 0.3))
    k3 = box(sim, "k3", (1.5, 0.0, 0.3))
    for step in range(60):
        yield
        if 20 <= step < 44:
            shift(k3, dx=-0.05)
        if 50 <= step < 56:
            shift(k3, dx=0.1)


@scene("link_contact", **dict(KINEMATIC, timestep=0.02))
def link_contact(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """An arm touches boxes with its links, not its base: the end effector rests
    against a box overhead; the elbow is placed bent so the upper arm reaches a
    box beside it, then driven straight again by kinematic joint motion."""
    arm = Agent.from_params(
        AgentSpawnParams(
            urdf_path="robots/arm_robot.urdf",
            initial_pose=Pose.from_xyz(0.0, 0.0, 0.0),
            use_fixed_base=True,
            collision_mode=CollisionMode.NORMAL_3D,
            name="arm",
        ),
        sim_core=sim,
    )
    box(sim, "overhead", (0.0, 0.0, 0.78), half=(0.05, 0.05, 0.05))
    box(sim, "beside", (0.0, -0.42, 0.35), half=(0.05, 0.05, 0.05))
    box(sim, "pedestal", (0.0, 0.0, -0.3), half=(0.3, 0.3, 0.2), mode=CollisionMode.STATIC)
    for step in range(40):
        yield
        if step == 4:
            arm.set_joint_position("shoulder_to_elbow", 1.57)
        if step == 14:
            arm.set_joint_target("shoulder_to_elbow", 0.0)


@scene("agents_crossing", **dict(KINEMATIC, spatial_hash_cell_size=2.0))
def agents_crossing(sim: MultiRobotSimulationCore) -> Iterator[None]:
    """Four robots drive through the same crossing, past a static obstacle."""
    box(sim, "obstacle", (1.0, 1.0, 0.2), mode=CollisionMode.STATIC)
    robots = [
        robot(sim, "r_west", (-3.0, 0.0), yaw=0.0),
        robot(sim, "r_east", (3.0, 0.0), yaw=3.14159),
        robot(sim, "r_south", (0.0, -3.0), yaw=1.5708),
        robot(sim, "r_north", (0.0, 3.0), yaw=-1.5708),
    ]
    goals = [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)]
    for agent, (x, y) in zip(robots, goals):
        go_to(agent, x, y)
    for _ in range(50):
        yield
