#!/usr/bin/env python3
"""
Demonstration of the WCE Transformation Algebra.

This script demonstrates the features of the transformation kernel:
- TransformationSpec as a first-class concept
- Explicit invariants
- Composition of transformations
- Topology
- TransformGraph
- MacroTransformation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import pi

print("=" * 70)
print("WCE TRANSFORMATION ALGEBRA DEMO")
print("=" * 70)
print()

# ============================================================================
# PART 1: Invariants
# ============================================================================
print("1. Fundamental invariants")
print("-" * 70)

from core.invariants import Invariant, InvariantSet, TRANSLATION_INVARIANTS, ROTATION_INVARIANTS

# Display a few invariants
print(f"Translation invariants: {TRANSLATION_INVARIANTS}")
print(f"  - Preserves distance: {TRANSLATION_INVARIANTS.preserves(Invariant.DISTANCE)}")
print(f"  - Preserves angle: {TRANSLATION_INVARIANTS.preserves(Invariant.ANGLE)}")
print(f"  - Preserves topology: {TRANSLATION_INVARIANTS.preserves_topology}")
print()

# ============================================================================
# PART 2: Primitive transformations
# ============================================================================
print("2. Primitive transformations")
print("-" * 70)

from core.transformations import (
    TransformationSpec,
    IdentityTransformation,
    CompositeTransformation,
    MacroTransformation,
    create_translation_transformation,
    create_rotation_transformation,
    create_scale_transformation,
)

# Creation of primitive transformations
identity = IdentityTransformation()
translation = create_translation_transformation(2.0, 0.0)
rotation = create_rotation_transformation(pi/4, "rotate(45°)")
scale_uniform = create_scale_transformation(2.0, 2.0)
scale_non_uniform = create_scale_transformation(2.0, 3.0)

print(f"Identity: {identity}")
print(f"  - Invertible: {identity.invertible}")
print(f"  - Preserves geometry: {identity.preserves_geometry()}")
print()

print(f"Translation (2, 0): {translation}")
print(f"  - Invertible: {translation.invertible}")
print(f"  - Preserves geometry: {translation.preserves_geometry()}")
print(f"  - Preserves topology: {translation.preserves_topology()}")
print()

print(f"Rotation (45°): {rotation}")
print(f"  - Invertible: {rotation.invertible}")
print(f"  - Preserves geometry: {rotation.preserves_geometry()}")
print(f"  - Preserves topology: {rotation.preserves_topology()}")
print()

print(f"Uniform scale (2, 2): {scale_uniform}")
print(f"  - Invertible: {scale_uniform.invertible}")
print(f"  - Preserves geometry: {scale_uniform.preserves_geometry()}")
print(f"  - Preserves topology: {scale_uniform.preserves_topology()}")
print()

print(f"Non-uniform scale (2, 3): {scale_non_uniform}")
print(f"  - Invertible: {scale_non_uniform.invertible}")
print(f"  - Preserves geometry: {scale_non_uniform.preserves_geometry()}")
print(f"  - Preserves topology: {scale_non_uniform.preserves_topology()}")
print()

# ============================================================================
# PART 3: Composition
# ============================================================================
print("3. Composition of transformations")
print("-" * 70)

# Composition with the @ operator
composed = translation @ rotation @ scale_uniform
print(f"Composition: {composed}")
print(f"  - Number of transformations: {composed.get_depth()}")
print(f"  - Preserves geometry: {composed.preserves_geometry()}")
print(f"  - Preserves topology: {composed.preserves_topology()}")
print(f"  - Invertible: {composed.invertible}")
print(f"  - Total cost: {composed.cost}")
print()

# Test the application
point = (1.0, 0.0)
print(f"Initial point: {point}")
print(f"After translation: {translation.apply(point)}")
print(f"After rotation: {rotation.apply(point)}")
print(f"After full composition: {composed.apply(point)}")
print()

# Test the inverse
print("Inversion of the composition:")
inverse_composed = composed.get_inverse()
if inverse_composed:
    print(f"  - Inverse exists: {inverse_composed.name}")
    transformed = composed.apply(point)
    inverted = inverse_composed.apply(transformed)
    print(f"  - Original: {point}")
    print(f"  - After composition: {transformed}")
    print(f"  - After inverse: {inverted}")
    print(f"  - Original recovered: {abs(inverted[0] - point[0]) < 1e-5 and abs(inverted[1] - point[1]) < 1e-5}")
print()

# ============================================================================
# PART 4: MacroTransformation
# ============================================================================
print("4. Macro-transformation (encapsulated operation)")
print("-" * 70)

# Create a macro for "place_on_table"
place_on_table_steps = [
    create_translation_transformation(0.0, 1.0),  # lift
    create_rotation_transformation(0.0, "reset_rotation"),  # null rotation
    create_translation_transformation(0.0, -1.0),  # lower onto the table
]

# Create the base composition
place_on_table_composite = place_on_table_steps[0] @ place_on_table_steps[1] @ place_on_table_steps[2]

# Create the macro
place_on_table = MacroTransformation(
    name="place_on_table",
    base_transformation=place_on_table_composite,
    description="Place un objet sur la table"
)

print(f"Macro: {place_on_table}")
print(f"  - Preserves topology: {place_on_table.preserves.preserves_topology}")
print(f"  - Preserves geometry: {place_on_table.preserves.preserves_geometry}")
print(f"  - Invertible: {place_on_table.invertible}")
print()

# ============================================================================
# PART 5: Topology
# ============================================================================
print("5. Topology and topological transformations")
print("-" * 70)

from core.topology import TopologySignature, DISK_TOPOLOGY, RING_TOPOLOGY, SEGMENT_TOPOLOGY
from core.transformations import CutTransformation, MergeTransformation

print(f"Disc: {DISK_TOPOLOGY}")
print(f"  - Is solid: {DISK_TOPOLOGY.is_solid()}")
print(f"  - Is simple: {DISK_TOPOLOGY.is_simple()}")
print()

print(f"Ring: {RING_TOPOLOGY}")
print(f"  - Is solid: {RING_TOPOLOGY.is_solid()}")
print(f"  - Has holes: {RING_TOPOLOGY.holes > 0}")
print()

# Apply a cut to a disc
cut = CutTransformation()
cut_topology = cut.apply(DISK_TOPOLOGY)
print(f"Disc -> after cut: {cut_topology}")
print(f"  - Connected components: {cut_topology.connected_components} (was 1)")
print()

# Apply a merge
merge = MergeTransformation()
merged_topology = merge.apply(cut_topology)
print(f"After merge: {merged_topology}")
print(f"  - Connected components: {merged_topology.connected_components}")
print()

# ============================================================================
# PART 6: TransformGraph
# ============================================================================
print("6. TransformGraph - History and tracking")
print("-" * 70)

from core.transform_graph import TransformGraph, TransformNode, create_linear_graph
from core.topology import DISK_TOPOLOGY

# Create a linear graph
graph = create_linear_graph(
    name="circle",
    transformations=[
        create_translation_transformation(2.0, 0.0),
        create_rotation_transformation(pi/4),
        create_scale_transformation(2.0, 2.0),
    ],
    initial_config="circle(x=0, y=0, r=1)",
    initial_topology=DISK_TOPOLOGY
)

print(f"Graph created: {graph}")
print()

# Display the history
summary = graph.get_history_summary()
print("History summary:")
for key, value in summary.items():
    print(f"  - {key}: {value}")
print()

# Display the path
path_names = graph.current.get_path_names()
print(f"Transformation path: {' -> '.join(path_names)}")
print()

# Check the invariants
current_invariants = graph.get_current_invariants()
print(f"Preserved invariants: {current_invariants}")
print(f"  - Preserves topology: {current_invariants.preserves_topology}")
print(f"  - Preserves geometry: {current_invariants.preserves_geometry}")
print()

# ============================================================================
# PART 7: Integrated demonstration with World
# ============================================================================
print("7. Integration with the existing system")
print("-" * 70)

from core.world import World, Entity
from core.transform import Transform2D

# Create a simple world
world = World()
world.add_entity(Entity(
    name="ball",
    kind="ball",
    properties={'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0, 'm': 1.0}
))

print(f"World created with entity: ball")
print()

# Create a Transform2D which is now a first-class transformation
T = Transform2D.translation(5.0, 0.0).compose(Transform2D.rotation(pi/2))
print(f"Transform2D: {T}")

# Apply to a point
p = (1.0, 0.0)
pt = T.transform_point(*p)
print(f"Point {p} -> {pt}")

# Check the inverse
T_inv = T.inverse()
ptt = T_inv.transform_point(*pt)
print(f"Inverse: {ptt} (original: {p})")
print(f"Precision: {abs(ptt[0] - p[0]) < 1e-10 and abs(ptt[1] - p[1]) < 1e-10}")
print()

# ============================================================================
# PART 8: Conceptual demonstration - Circle to Ellipse
# ============================================================================
print("8. Deformation: Circle -> Ellipse")
print("-" * 70)

from core.deformation import StretchDeformation

# Create a stretch deformation
stretch = StretchDeformation(stretch_x=2.0, stretch_y=3.0)
print(f"Deformation: {stretch.name}")
print(f"  - Preserves topology: {stretch.topology_preservation.preserves_connected_components}")
print(f"  - Preserves holes: {stretch.topology_preservation.preserves_holes}")
print(f"  - Preserves distances: False (stretch modifies the geometry)")
print()

# Apply to a conceptual circle
print("Circle (r=1) -> Ellipse (via stretch(2,3))")
print("  - Initial topology: DISK (1 component, 0 holes)")
print("  - Final topology: DISK (unchanged)")
print("  - Geometry modified: Yes")
print("  - Preserved invariants: connectivity, holes, dimension")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 70)
print("✓ FUNDAMENTAL PROPERTIES DEMONSTRATED")
print("=" * 70)
print()
print("1. Small set of primitive transformations")
print("   + composition")
print("   = large number of behaviours")
print()
print("2. Explicit invariants for each transformation")
print("   → makes it possible to reason about what changes and what remains")
print()
print("3. Topology separated from geometry")
print("   → makes it possible to distinguish the fundamental properties")
print()
print("4. TransformGraph to track the history")
print("   → makes it possible to answer: 'Which transformations produced this state?'")
print()
print("5. MacroTransformation to encapsulate frequent compositions")
print("   → makes it possible to create new operations without modifying the code")
print()
print("=" * 70)
print("✓ TRANSFORMATION ALGEBRA DEMO COMPLETE")
print("=" * 70)
