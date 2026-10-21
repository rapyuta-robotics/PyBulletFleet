#!/usr/bin/env python3
"""Download Gazebo Fuel models referenced by rmf_demos world files.

Scans all .world files in the rmf_demos_maps install directory for
``https://fuel.gazebosim.org/...`` URIs and downloads each unique model
into the local Fuel cache (``~/.gz/fuel/...``).

This eliminates the need to run Gazebo to populate the cache — PyBulletFleet's
``sdf_loader`` resolves Fuel URIs from this same cache.

Usage (during Docker build)::

    python3 docker/download_fuel_models.py

Environment variables:
    GZ_FUEL_CACHE_DIR      Override cache directory (default: ``~/.gz/fuel``)
    FUEL_MAPS_DIR          Override maps search directory
    ALLOW_NO_FUEL_MODELS   Set to ``1`` to accept worlds that reference no Fuel
                           models (an empty cache) instead of failing. Missing
                           world files fail regardless.

The cache directory is always created, so a Dockerfile that copies it never
fails on an empty result. Finding no world files, or worlds that name no Fuel
models, fails the build with an explanation instead: the bridge's demo worlds
then load without their meshes, which is not what the image is for.
"""

import glob
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

FUEL_CACHE = os.environ.get("GZ_FUEL_CACHE_DIR", os.path.expanduser("~/.gz/fuel"))
MAPS_DIR = os.environ.get(
    "FUEL_MAPS_DIR",
    "/rmf_demos_ws/install/rmf_demos_maps/share/rmf_demos_maps/maps",
)


def find_fuel_models(maps_dir: str) -> dict[tuple[str, str], str]:
    """Scan all .world files for Fuel model URIs, deduplicated to (owner, model_name).

    Returns dict mapping ``(owner, model_name)`` → canonical model URL.
    """
    models: dict[tuple[str, str], str] = {}
    for world_file in sorted(glob.glob(os.path.join(maps_dir, "**", "*.world"), recursive=True)):
        try:
            tree = ET.parse(world_file)
            for uri_el in tree.iter("uri"):
                text = (uri_el.text or "").strip()
                if "fuel.gazebosim.org" not in text:
                    continue
                owner, name = parse_fuel_url(text)
                if owner and name:
                    models[(owner, name)] = f"https://fuel.gazebosim.org/1.0/{owner}/models/{name}"
        except Exception as e:
            print(f"  WARN: Could not parse {world_file}: {e}", file=sys.stderr)
    return models


def parse_fuel_url(url: str) -> tuple[str | None, str | None]:
    """Parse Fuel URL into (owner, model_name).

    Handles both model-level and file-level URIs::

        https://fuel.gazebosim.org/1.0/OpenRobotics/models/Cafe table
        https://fuel.gazebosim.org/1.0/OpenRobotics/models/Chair/2/files/meshes/Chair.dae

    Returns only the owner and model name (not version or file paths).
    """
    # Strip https://fuel.gazebosim.org/1.0/
    rest = url.replace("https://fuel.gazebosim.org/1.0/", "")
    parts = rest.split("/")
    # parts: [owner, "models", name, ...]
    if len(parts) >= 3 and parts[1] == "models":
        return parts[0], parts[2]  # owner, model_name only
    return None, None


def download_model(owner: str, model_name: str) -> bool:
    """Download a single Fuel model and extract to cache.

    Cache structure: ~/.gz/fuel/fuel.gazebosim.org/<owner_lower>/models/<name_lower>/<version>/
    """
    api_url = f"https://fuel.gazebosim.org/1.0/{owner}/models/{urllib.parse.quote(model_name)}"

    # Get model metadata (version number)
    try:
        req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        meta = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: Failed to fetch metadata for {model_name}: {e}", file=sys.stderr)
        return False

    version = meta.get("version", 1)

    # Cache directory (lowercase owner and model name, matching Gazebo convention)
    cache_dir = os.path.join(
        FUEL_CACHE,
        "fuel.gazebosim.org",
        owner.lower(),
        "models",
        model_name.lower(),
        str(version),
    )

    # Skip if already cached
    if os.path.isfile(os.path.join(cache_dir, "model.sdf")):
        print(f"  CACHED: {model_name} v{version}")
        return True

    # Download ZIP
    # Fuel API: GET /1.0/<owner>/models/<name>/<version>/<name>.zip
    encoded_name = urllib.parse.quote(model_name)
    zip_url = f"https://fuel.gazebosim.org/1.0/{owner}/models/{encoded_name}/{version}/{encoded_name}.zip"
    try:
        resp = urllib.request.urlopen(zip_url, timeout=60)
        zip_data = resp.read()
    except Exception as e:
        print(f"  ERROR: Failed to download {model_name}: {e}", file=sys.stderr)
        return False

    # Extract to cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(zip_data))
        z.extractall(cache_dir)
    except Exception as e:
        print(f"  ERROR: Failed to extract {model_name}: {e}", file=sys.stderr)
        return False

    print(f"  OK: {model_name} v{version} ({len(zip_data)} bytes)")

    # Create texture symlinks in meshes/ so PyBullet can resolve DAE <init_from> refs.
    # Gazebo Fuel models store textures in materials/textures/ but DAE files reference
    # them by bare filename — PyBullet only looks relative to the DAE file's directory.
    _symlink_textures(cache_dir)

    return True


def _symlink_textures(model_version_dir: str) -> None:
    """Symlink texture images into meshes/ dirs for PyBullet DAE texture resolution.

    Fuel model layout:
        <ver>/meshes/model.dae          ← DAE references "texture.jpg"
        <ver>/materials/textures/texture.jpg

    PyBullet resolves relative to the DAE → needs textures in meshes/.
    We create symlinks: meshes/texture.jpg → ../materials/textures/texture.jpg
    """
    tex_dir = os.path.join(model_version_dir, "materials", "textures")
    if not os.path.isdir(tex_dir):
        return

    meshes_dir = os.path.join(model_version_dir, "meshes")
    if not os.path.isdir(meshes_dir):
        return

    for fname in os.listdir(tex_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
            continue
        link_path = os.path.join(meshes_dir, fname)
        if os.path.exists(link_path):
            continue  # already exists (symlink or real file)
        # Use relative symlink for portability within the image
        target = os.path.join("..", "materials", "textures", fname)
        try:
            os.symlink(target, link_path)
        except OSError:
            pass  # best-effort; skip on failure


def count_model_uris(maps_dir: str) -> int:
    """Count ``model://<name>`` URIs in the world files: models referenced by name only."""
    count = 0
    for world_file in glob.glob(os.path.join(maps_dir, "**", "*.world"), recursive=True):
        try:
            count += sum(
                1 for uri_el in ET.parse(world_file).iter("uri") if (uri_el.text or "").strip().startswith("model://")
            )
        except Exception:
            pass
    return count


def main():
    # Always leave the cache directory behind, even empty, so that copying it
    # out of a build stage cannot fail on a missing path.
    os.makedirs(FUEL_CACHE, exist_ok=True)
    allow_empty = os.environ.get("ALLOW_NO_FUEL_MODELS", "") == "1"

    print(f"Scanning world files in {MAPS_DIR} ...")
    worlds = glob.glob(os.path.join(MAPS_DIR, "**", "*.world"), recursive=True)
    if not worlds:
        # Always fatal, override or not: an image without its demo worlds is the
        # base-image regression this check exists to catch.
        print(f"ERROR: no .world files under {MAPS_DIR}. Has the base image moved its maps?", file=sys.stderr)
        return 1

    models = find_fuel_models(MAPS_DIR)
    print(f"Found {len(models)} unique Fuel models in {len(worlds)} world files")

    if not models:
        by_name = count_model_uris(MAPS_DIR)
        print(
            f"ERROR: the world files name no Fuel models ({by_name} model:// URIs instead). "
            "The base image's worlds no longer reference Fuel URLs, so there is nothing to "
            "cache; pin a base image whose worlds do, or set ALLOW_NO_FUEL_MODELS=1.",
            file=sys.stderr,
        )
        return 0 if allow_empty else 1

    print(f"Downloading {len(models)} models to {FUEL_CACHE} ...")
    ok, fail = 0, 0
    for owner, name in sorted(models.keys(), key=lambda x: x[1].lower()):
        if download_model(owner, name):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed")
    if fail:
        print("WARNING: Some models failed to download. They will appear as missing meshes.", file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
