#!/usr/bin/env python3
"""
generate_markers_and_signs.py

Reads the building YAML, finds aruco_* and direction_* vertices,
generates textures, and injects inline SDF <model> blocks into
the output.world file.

Usage:
  python3 generate_markers_and_signs.py <building.yaml> <output.world> <worlds_dir>
"""

import sys
import os
import math
import yaml
import cv2
import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IMG_PX = 512            # generated marker image resolution
ARUCO_BOX_SIZE = 0.5          # metres
ARUCO_BOX_COLOR = "0.9 0.6 0.1 1"  # distinct orange-amber colour

SIGN_PANEL_W = 0.3            # metres
SIGN_PANEL_H = 0.3
SIGN_PANEL_THICKNESS = 0.015
SIGN_PANEL_ELEVATION = 0.30   # centre-of-panel above ground
POLE_RADIUS = 0.015
POLE_HEIGHT = SIGN_PANEL_ELEVATION - SIGN_PANEL_H / 2  # ground to bottom of panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_scale(vertices, measurements):
    """Compute pixels-per-metre from the first measurement line."""
    m = measurements[0]
    idx_a, idx_b = m[0], m[1]
    dist_m = m[2]["distance"][1]
    va = vertices[idx_a]
    vb = vertices[idx_b]
    px_dist = math.hypot(vb[0] - va[0], vb[1] - va[1])
    return px_dist / dist_m


def px_to_world(px_x, px_y, scale):
    """Convert pixel coords to Gazebo world coords (metres)."""
    return px_x / scale, -px_y / scale


def deg_to_rad(d):
    return d * math.pi / 180.0


# ---------------------------------------------------------------------------
# ArUco generation
# ---------------------------------------------------------------------------
def generate_aruco_image(marker_id, out_path):
    """Generate a white-bordered ArUco marker PNG."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_core = cv2.aruco.drawMarker(aruco_dict, marker_id, ARUCO_IMG_PX - 80)
    # white border
    img = 255 * np.ones((ARUCO_IMG_PX, ARUCO_IMG_PX), dtype=np.uint8)
    off = 40
    img[off:off + marker_core.shape[0], off:off + marker_core.shape[1]] = marker_core
    cv2.imwrite(out_path, img)


def aruco_model_sdf(name, wx, wy, yaw_rad, texture_path):
    """Return an inline SDF <model> string for an ArUco box."""
    half = ARUCO_BOX_SIZE / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{wx} {wy} {half} 0 0 {yaw_rad}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE}</size></box></geometry>
          <material>
            <ambient>{ARUCO_BOX_COLOR}</ambient>
            <diffuse>{ARUCO_BOX_COLOR}</diffuse>
            <pbr>
              <metal>
                <albedo_map>{texture_path}</albedo_map>
              </metal>
            </pbr>
          </material>
        </visual>
        <collision name="collision">
          <geometry><box><size>{ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE}</size></box></geometry>
        </collision>
      </link>
    </model>"""


# ---------------------------------------------------------------------------
# Sign generation
# ---------------------------------------------------------------------------
def generate_sign_image(sign_type, sign_config, worlds_dir, out_path):
    """Generate a transformed sign PNG based on config."""
    cfg = sign_config[sign_type]
    base_path = os.path.join(worlds_dir, cfg["base_image"])
    img = Image.open(base_path).convert("RGBA")

    rot = cfg.get("rotation_cw_deg", 0)
    if rot:
        # PIL rotate is CCW, so negate
        img = img.rotate(-rot, expand=True, resample=Image.BICUBIC)

    if cfg.get("mirror_horizontal", False):
        img = ImageOps.mirror(img)

    img.save(out_path)


def sign_model_sdf(name, wx, wy, yaw_rad, sign_texture, wood_texture):
    """Return an inline SDF <model> for a signboard on a pole."""
    panel_z = SIGN_PANEL_ELEVATION
    pole_z = POLE_HEIGHT / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{wx} {wy} 0 0 0 {yaw_rad}</pose>
      <link name="link">
        <!-- Pole -->
        <visual name="pole">
          <pose>0 0 {pole_z} 0 0 0</pose>
          <geometry><cylinder><radius>{POLE_RADIUS}</radius><length>{POLE_HEIGHT}</length></cylinder></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>
            <pbr><metal><albedo_map>file://{wood_texture}</albedo_map></metal></pbr>
          </material>
        </visual>
        <collision name="pole_col">
          <pose>0 0 {pole_z} 0 0 0</pose>
          <geometry><cylinder><radius>{POLE_RADIUS}</radius><length>{POLE_HEIGHT}</length></cylinder></geometry>
        </collision>
        <!-- Sign panel -->
        <visual name="panel">
          <pose>0 0 {panel_z} 0 0 0</pose>
          <geometry><box><size>{SIGN_PANEL_THICKNESS} {SIGN_PANEL_W} {SIGN_PANEL_H}</size></box></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>
            <pbr><metal><albedo_map>file://{wood_texture}</albedo_map></metal></pbr>
          </material>
        </visual>
        <!-- Sign face (front) -->
        <visual name="sign_front">
          <pose>{SIGN_PANEL_THICKNESS / 2 + 0.001} 0 {panel_z} 0 0 0</pose>
          <geometry><box><size>0.001 {SIGN_PANEL_W} {SIGN_PANEL_H}</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            <pbr><metal><albedo_map>file://{sign_texture}</albedo_map></metal></pbr>
          </material>
        </visual>
        <collision name="panel_col">
          <pose>0 0 {panel_z} 0 0 0</pose>
          <geometry><box><size>{SIGN_PANEL_THICKNESS} {SIGN_PANEL_W} {SIGN_PANEL_H}</size></box></geometry>
        </collision>
      </link>
    </model>"""


# ---------------------------------------------------------------------------
# Vertex name parsers
# ---------------------------------------------------------------------------
def parse_aruco(name):
    """Parse 'aruco_{id}_{yaw}' → (marker_id, yaw_deg)."""
    parts = name.split("_")  # ['aruco', id, yaw]
    marker_id = int(parts[1])
    yaw_deg = float(parts[-1])
    return marker_id, yaw_deg


def parse_direction(name):
    """Parse 'direction_{id}_{type}_{yaw}' → (sign_id, sign_type, yaw_deg).
    Type may contain underscores (e.g. rotate_180_clockwise).
    Yaw is always the last _-token."""
    parts = name.split("_")
    # parts[0] = 'direction'
    sign_id = parts[1]
    yaw_deg = float(parts[-1])
    sign_type = "_".join(parts[2:-1])
    return sign_id, sign_type, yaw_deg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <building.yaml> <output.world> <worlds_dir>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    world_path = sys.argv[2]
    worlds_dir = sys.argv[3]

    # Output directory for generated textures
    gen_dir = os.path.join(worlds_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    # Load building YAML
    with open(yaml_path, "r") as f:
        building = yaml.safe_load(f)

    # Load sign config
    sign_config_path = os.path.join(worlds_dir, "sign_config.yaml")
    with open(sign_config_path, "r") as f:
        sign_config = yaml.safe_load(f)

    # Get the first level (we only have floor_0)
    level_name = list(building["levels"].keys())[0]
    level = building["levels"][level_name]
    vertices = level["vertices"]
    measurements = level.get("measurements", [])

    if not measurements:
        print("ERROR: No measurements found — cannot compute scale.", file=sys.stderr)
        sys.exit(1)

    scale = compute_scale(vertices, measurements)
    print(f"[markers_signs] Scale: {scale:.4f} px/m")

    wood_texture = os.path.join(worlds_dir, "wood_texture.jpg")

    sdf_blocks = []

    # --- ArUco markers ---
    for idx, v in enumerate(vertices):
        name = v[3] if len(v) >= 4 else ""
        if not isinstance(name, str) or not name.startswith("aruco_"):
            continue

        marker_id, yaw_deg = parse_aruco(name)
        wx, wy = px_to_world(v[0], v[1], scale)
        yaw_rad = deg_to_rad(yaw_deg)

        tex_path = os.path.join(gen_dir, f"aruco_{marker_id}.png")
        generate_aruco_image(marker_id, tex_path)

        model_name = f"aruco_{marker_id}_v{idx}"
        sdf = aruco_model_sdf(model_name, wx, wy, yaw_rad, tex_path)
        sdf_blocks.append(sdf)
        print(f"  [aruco] id={marker_id} yaw={yaw_deg}° @ ({wx:.2f}, {wy:.2f})")

    # --- Direction signs ---
    for idx, v in enumerate(vertices):
        name = v[3] if len(v) >= 4 else ""
        if not isinstance(name, str) or not name.startswith("direction_"):
            continue

        sign_id, sign_type, yaw_deg = parse_direction(name)
        wx, wy = px_to_world(v[0], v[1], scale)
        yaw_rad = deg_to_rad(yaw_deg)

        if sign_type not in sign_config:
            print(f"  [sign] WARNING: unknown type '{sign_type}' in vertex '{name}', skipping",
                  file=sys.stderr)
            continue

        tex_path = os.path.join(gen_dir, f"sign_{sign_id}_{sign_type}.png")
        generate_sign_image(sign_type, sign_config, worlds_dir, tex_path)

        model_name = f"sign_{sign_id}_{sign_type}_v{idx}"
        sdf = sign_model_sdf(
            model_name, wx, wy, yaw_rad,
            tex_path, wood_texture)
        sdf_blocks.append(sdf)
        print(f"  [sign] id={sign_id} type={sign_type} yaw={yaw_deg}° @ ({wx:.2f}, {wy:.2f})")

    if not sdf_blocks:
        print("[markers_signs] No markers or signs found.")
        return

    # --- Inject into output.world ---
    with open(world_path, "r") as f:
        world_xml = f.read()

    injection = "\n".join(sdf_blocks)
    # Insert just before </world>
    marker_tag = "</world>"
    if marker_tag not in world_xml:
        print("ERROR: Could not find </world> in output.world", file=sys.stderr)
        sys.exit(1)

    world_xml = world_xml.replace(marker_tag, injection + "\n  " + marker_tag)

    with open(world_path, "w") as f:
        f.write(world_xml)

    print(f"[markers_signs] Injected {len(sdf_blocks)} models into {world_path}")


if __name__ == "__main__":
    main()
