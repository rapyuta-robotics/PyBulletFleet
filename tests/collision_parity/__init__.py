"""Collision parity: what the collision system reports over scripted scenes.

A scene (``scenes.py``) builds a simulation and drives it step by step. The
recorder (``trace.py``) writes down, for every step, the ``COLLISION_STARTED``
and ``COLLISION_ENDED`` events, the active pairs of the collision observation
and each pair's ``signed_distance``. The traces recorded on a reference
revision are kept under ``golden/`` and ``tests/test_collision_parity.py``
compares a fresh recording against them, so that a rewrite of the collision
pipeline can show it reports the same thing, and say in one place where it
knowingly differs.

Record the goldens again with ``python -m tests.collision_parity.record``.
"""
