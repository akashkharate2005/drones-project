"""Procedural generator for Webots crowd surveillance world (crowd_plaza.wbt).

Places the Mavic 2 Pro drone, 30-60 animated Pedestrians with randomized trajectory waypoints,
and portrait billboards for facial inspection in an open cobblestone plaza.
"""
import argparse
import math
import os
import random
from typing import List, Tuple


def generate_webots_world(
    crowd_size: int = 45,
    density: str = "medium",
    output_path: str = "worlds/crowd_plaza.wbt"
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(42)

    plaza_radius = 28.0
    if density == "dense":
        spawn_radius = 16.0
    elif density == "sparse":
        spawn_radius = 26.0
    else: # medium
        spawn_radius = 22.0

    # Ensure billboard image paths are referenced relative to worlds/
    billboards = [
        {"name": "BILLBOARD_ALICE", "x": 6.0, "z": 8.0, "rot": 0.0, "tex": "../data/billboards/billboard_alice.jpg"},
        {"name": "BILLBOARD_BOB", "x": -10.0, "z": -6.0, "rot": 1.57, "tex": "../data/billboards/billboard_bob.jpg"},
        {"name": "BILLBOARD_CHARLIE", "x": 12.0, "z": -14.0, "rot": -1.57, "tex": "../data/billboards/billboard_charlie.jpg"}
    ]

    lines = []
    # Header and declarations
    lines.append('#VRML_SIM R2023b utf-8')
    lines.append('')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/dji/mavic/protos/Mavic2Pro.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/humans/pedestrian/protos/Pedestrian.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"')
    lines.append('')
    lines.append('WorldInfo {')
    lines.append('  info [')
    lines.append('    "Autonomous Drone Smart Crowd Monitoring Surveillance World"')
    lines.append('    "Mavic 2 Pro with EXTERN controller inspecting crowd plaza and portrait billboards"')
    lines.append('  ]')
    lines.append('  title "Smart Crowd Monitoring Plaza"')
    lines.append('  basicTimeStep 32')
    lines.append('  coordinateSystem "NUE"')
    lines.append('}')
    lines.append('')
    lines.append('Viewpoint {')
    lines.append('  orientation -0.57 0.70 0.42 1.3')
    lines.append('  position 35 45 40')
    lines.append('  near 0.2')
    lines.append('}')
    lines.append('')
    lines.append('TexturedBackground {}')
    lines.append('TexturedBackgroundLight {}')
    lines.append('')
    # Plaza Arena (60m x 60m)
    lines.append('RectangleArena {')
    lines.append('  translation 0 0 0')
    lines.append('  floorSize 60 60')
    lines.append('  floorTileSize 2 2')
    lines.append('  wallHeight 2')
    lines.append('}')
    lines.append('')
    # Central Monument / Fountain
    lines.append('Solid {')
    lines.append('  translation 0 0.8 0')
    lines.append('  children [')
    lines.append('    Shape {')
    lines.append('      appearance PBRAppearance {')
    lines.append('        baseColor 0.7 0.6 0.5')
    lines.append('        roughness 0.6')
    lines.append('      }')
    lines.append('      geometry Cylinder {')
    lines.append('        height 1.6')
    lines.append('        radius 3.0')
    lines.append('      }')
    lines.append('    }')
    lines.append('  ]')
    lines.append('  boundingObject Cylinder {')
    lines.append('    height 1.6')
    lines.append('    radius 3.0')
    lines.append('  }')
    lines.append('}')
    lines.append('')

    # Portrait Billboards for Close Inspection
    for b in billboards:
        lines.append(f'DEF {b["name"]} Solid {{')
        lines.append(f'  translation {b["x"]} 1.6 {b["z"]}')
        lines.append(f'  rotation 0 1 0 {b["rot"]}')
        lines.append('  children [')
        # Billboard support post
        lines.append('    Transform {')
        lines.append('      translation 0 -0.8 0')
        lines.append('      children [')
        lines.append('        Shape {')
        lines.append('          appearance PBRAppearance { baseColor 0.3 0.3 0.3 }')
        lines.append('          geometry Cylinder { height 1.6 radius 0.08 }')
        lines.append('        }')
        lines.append('      ]')
        lines.append('    }')
        # Billboard display panel with portrait texture
        lines.append('    Shape {')
        lines.append('      appearance PBRAppearance {')
        lines.append('        baseColorMap ImageTexture {')
        lines.append(f'          url [ "{b["tex"]}" ]')
        lines.append('        }')
        lines.append('        roughness 0.4')
        lines.append('      }')
        lines.append('      geometry Box { size 2.0 1.8 0.1 }')
        lines.append('    }')
        lines.append('  ]')
        lines.append('  boundingObject Box { size 2.0 3.2 0.2 }')
        lines.append('}')
        lines.append('')

    # Procedural Animated Pedestrians
    lines.append(f'# {crowd_size} Procedural Pedestrians with Randomized Waypoint Trajectories')
    for i in range(1, crowd_size + 1):
        x = round(random.uniform(-spawn_radius, spawn_radius), 2)
        z = round(random.uniform(-spawn_radius, spawn_radius), 2)
        # Avoid spawning directly inside central monument
        if math.hypot(x, z) < 3.5:
            x += 4.5

        # Create randomized looped patrol trajectory
        wp1_x = round(random.uniform(-spawn_radius, spawn_radius), 2)
        wp1_z = round(random.uniform(-spawn_radius, spawn_radius), 2)
        wp2_x = round(random.uniform(-spawn_radius, spawn_radius), 2)
        wp2_z = round(random.uniform(-spawn_radius, spawn_radius), 2)
        wp3_x = round(random.uniform(-spawn_radius, spawn_radius), 2)
        wp3_z = round(random.uniform(-spawn_radius, spawn_radius), 2)

        lines.append(f'DEF PEDESTRIAN_{i} Pedestrian {{')
        lines.append(f'  translation {x} 0 {z}')
        lines.append(f'  rotation 0 1 0 {round(random.uniform(-3.14, 3.14), 2)}')
        lines.append(f'  waypoints [')
        lines.append(f'    {wp1_x} 0 {wp1_z}')
        lines.append(f'    {wp2_x} 0 {wp2_z}')
        lines.append(f'    {wp3_x} 0 {wp3_z}')
        lines.append(f'    {x} 0 {z}')
        lines.append(f'  ]')
        lines.append('  controller "<none>"')
        lines.append('}')
        lines.append('')

    # Drone: DJI Mavic 2 Pro with EXTERN controller
    lines.append('# DJI Mavic 2 Pro Drone Payload')
    lines.append('DEF MAVIC_2_PRO Mavic2Pro {')
    lines.append('  translation 0 0.15 0')
    lines.append('  rotation 0 1 0 0')
    lines.append('  controller "<extern>"')
    lines.append('  supervisor TRUE')
    lines.append('  cameraSlot [')
    lines.append('    Camera {')
    lines.append('      width 960')
    lines.append('      height 720')
    lines.append('      fieldOfView 1.45')
    lines.append('    }')
    lines.append('  ]')
    lines.append('}')
    lines.append('')

    content = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Generated Webots world file: {output_path} ({crowd_size} pedestrians, density='{density}')")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Webots crowd surveillance world.")
    parser.add_argument("--crowd-size", type=int, default=45, help="Number of pedestrians (30-60)")
    parser.add_argument("--density", type=str, default="medium", choices=["sparse", "medium", "dense"])
    parser.add_argument("--output", type=str, default="worlds/crowd_plaza.wbt")
    args = parser.parse_args()

    generate_webots_world(crowd_size=args.crowd_size, density=args.density, output_path=args.output)
