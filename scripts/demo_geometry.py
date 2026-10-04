#!/usr/bin/env python3
"""
Démonstration du noyau géométrique WCE.

Ce script démontre les fonctionnalités du noyau géométrique :
- Transform2D
- PlaneCollider
- Topology
- Deformation
- PlaneCollision
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import pi

print("=" * 60)
print("WCE Geometry Kernel Demo")
print("=" * 60)
print()

# 1. Transform2D
print("1. Transform2D : translation, rotation, scale, compose, inverse")
print("-" * 60)

from core.transform import Transform2D

# Identité
print(f"Identité: {Transform2D.identity()}")

# Translation
print(f"Translation (5,10): {Transform2D.translation(5.0, 10.0)}")

# Rotation 90°
print(f"Rotation 90°: {Transform2D.rotation(pi/2)}")

# Scale
print(f"Scale (2,3): {Transform2D.scale(2.0, 3.0)}")

# Composition
T1 = Transform2D.translation(5.0, 0.0)
T2 = Transform2D.rotation(pi/2)
T3 = T2.compose(T1)
print(f"T2 ∘ T1: {T3}")

# Test inverse
T = Transform2D.translation(10.0, 5.0).compose(Transform2D.rotation(pi/4))
T_inv = T.inverse()
p = (3.0, 4.0)
pt = T.transform_point(*p)
ptt = T_inv.transform_point(*pt)
print(f"p={p} -> T(p)={pt} -> T^-1(T(p))={ptt}")
assert abs(ptt[0] - p[0]) < 1e-5 and abs(ptt[1] - p[1]) < 1e-5
print("✓ Inverse fonctionne")

print()

# 2. PlaneCollider
print("2. PlaneCollider : mur, sol, plan incliné")
print("-" * 60)

from physics.colliders import PlaneCollider

# Mur vertical droit à x=5
wall = PlaneCollider(
    normal_x=1.0,
    normal_y=0.0,
    transform=Transform2D.translation(5.0, 0.0)
)
print(f"Mur à x=5 : is_behind(6,0)={wall.is_behind(6.0, 0.0)}, is_behind(4,0)={wall.is_behind(4.0, 0.0)}")

# Sol horizontal à y=0
floor = PlaneCollider(
    normal_x=0.0,
    normal_y=-1.0,
    transform=Transform2D.identity()
)
print(f"Sol à y=0 : is_behind(0,1)={floor.is_behind(0.0, 1.0)}, is_behind(0,-1)={floor.is_behind(0.0, -1.0)}")

# Plan incliné à 45°
inclined = PlaneCollider(
    normal_x=0.0,
    normal_y=1.0,
    transform=Transform2D.rotation(pi/4)
)
nx, ny = inclined.world_normal()
print(f"Plan incliné : normale monde=({nx:.3f}, {ny:.3f})")
print(f"  is_behind(0,10)={inclined.is_behind(0.0, 10.0)} (y > x)")
print(f"  is_behind(10,0)={inclined.is_behind(10.0, 0.0)} (y < x)")

print()

# 3. Topology
print("3. Topology : signatures et préservation")
print("-" * 60)

from core.topology import (
    TopologySignature, DISK_TOPOLOGY, RING_TOPOLOGY, SEGMENT_TOPOLOGY,
    TopologyPreservation, get_topology_preservation_rotation,
    get_topology_preservation_uniform_scale
)

print(f"Disque : {DISK_TOPOLOGY}")
print(f"Anneau : {RING_TOPOLOGY}")
print(f"Segment : {SEGMENT_TOPOLOGY}")

rot_pres = get_topology_preservation_rotation()
print(f"Rotation préserve tout : {rot_pres.preserves_connected_components and rot_pres.preserves_holes and rot_pres.preserves_boundary and rot_pres.preserves_orientation}")

print()

# 4. Deformation
print("4. Deformation : stretch, identity, composite")
print("-" * 60)

from core.deformation import (
    StretchDeformation, IdentityDeformation, compose_deformations,
    circle_to_ellipse, square_to_rectangle
)

# Étirement
stretch = StretchDeformation(stretch_x=2.0, stretch_y=3.0)
print(f"Stretch (2,3) : {stretch.name}")
print(f"  Topology : {stretch.topology_preservation}")

# Cercle vers ellipse
ellipse = circle_to_ellipse(2.0, 3.0)
print(f"Circle -> Ellipse : {ellipse.name}")

# Composition
comp = compose_deformations(stretch, IdentityDeformation())
print(f"Composite : {comp.name}")

print()

# 5. Collision avec PlaneCollider
print("5. Collision avec PlaneCollider")
print("-" * 60)

from core.world import World, Entity
from core.transformation import Transformation, Change
from physics.colliders import Material
from physics.geometry_collision import PlaneCollision

world = World()

# Balle
world.add_entity(Entity(
    name='balle',
    kind='ball',
    properties={'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'm': 1.0}
))

# Mur droit avec PlaneCollider
world.add_entity(Entity(
    name='mur',
    kind='wall',
    properties={
        'collider': PlaneCollider(
            normal_x=1.0,
            normal_y=0.0,
            transform=Transform2D.translation(5.0, 0.0)
        ),
        'material': Material(restitution=0.0)
    }
))

# Déplacer la balle vers la droite
transformation = Transformation(
    name='move',
    changes=[
        Change('balle', 'x', 0.0, 10.0),
        Change('balle', 'vx', 10.0, 10.0),
    ]
)

plane_collision = PlaneCollision()
resolved = plane_collision.resolve(world, transformation)

x_final = [c.after for c in resolved.changes if c.property == 'x'][0]
vx_final = [c.after for c in resolved.changes if c.property == 'vx'][0]

print(f"Collision avec mur à x=5 :")
print(f"  Position finale : x={x_final:.1f} (attendu: 5.0)")
print(f"  Vitesse finale : vx={vx_final:.1f} (attendu: 0.0)")
print(f"  Collision détectée : {'+plane_collision' in resolved.name}")

print()
print("=" * 60)
print("✓ Démo géométrie terminée")
print("=" * 60)
