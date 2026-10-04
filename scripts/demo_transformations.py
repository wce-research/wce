#!/usr/bin/env python3
"""
Démonstration du WCE Transformation Algebra.

Ce script démontre les fonctionnalités du noyau de transformation :
- TransformationSpec comme concept de premier ordre
- Invariants explicites
- Composition de transformations
- Topologie
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
# PARTIE 1 : Invariants
# ============================================================================
print("1. Invariants fondamentaux")
print("-" * 70)

from core.invariants import Invariant, InvariantSet, TRANSLATION_INVARIANTS, ROTATION_INVARIANTS

# Afficher quelques invariants
print(f"Invariants de translation : {TRANSLATION_INVARIANTS}")
print(f"  - Préserve la distance : {TRANSLATION_INVARIANTS.preserves(Invariant.DISTANCE)}")
print(f"  - Préserve l'angle : {TRANSLATION_INVARIANTS.preserves(Invariant.ANGLE)}")
print(f"  - Préserve la topologie : {TRANSLATION_INVARIANTS.preserves_topology}")
print()

# ============================================================================
# PARTIE 2 : Transformations primitives
# ============================================================================
print("2. Transformations primitives")
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

# Création de transformations primitives
identity = IdentityTransformation()
translation = create_translation_transformation(2.0, 0.0)
rotation = create_rotation_transformation(pi/4, "rotate(45°)")
scale_uniform = create_scale_transformation(2.0, 2.0)
scale_non_uniform = create_scale_transformation(2.0, 3.0)

print(f"Identité : {identity}")
print(f"  - Invertible : {identity.invertible}")
print(f"  - Préserve la géométrie : {identity.preserves_geometry()}")
print()

print(f"Translation (2, 0) : {translation}")
print(f"  - Invertible : {translation.invertible}")
print(f"  - Préserve la géométrie : {translation.preserves_geometry()}")
print(f"  - Préserve la topologie : {translation.preserves_topology()}")
print()

print(f"Rotation (45°) : {rotation}")
print(f"  - Invertible : {rotation.invertible}")
print(f"  - Préserve la géométrie : {rotation.preserves_geometry()}")
print(f"  - Préserve la topologie : {rotation.preserves_topology()}")
print()

print(f"Scale uniforme (2, 2) : {scale_uniform}")
print(f"  - Invertible : {scale_uniform.invertible}")
print(f"  - Préserve la géométrie : {scale_uniform.preserves_geometry()}")
print(f"  - Préserve la topologie : {scale_uniform.preserves_topology()}")
print()

print(f"Scale non-uniforme (2, 3) : {scale_non_uniform}")
print(f"  - Invertible : {scale_non_uniform.invertible}")
print(f"  - Préserve la géométrie : {scale_non_uniform.preserves_geometry()}")
print(f"  - Préserve la topologie : {scale_non_uniform.preserves_topology()}")
print()

# ============================================================================
# PARTIE 3 : Composition
# ============================================================================
print("3. Composition de transformations")
print("-" * 70)

# Composition avec l'opérateur @
composed = translation @ rotation @ scale_uniform
print(f"Composition : {composed}")
print(f"  - Nombre de transformations : {composed.get_depth()}")
print(f"  - Préserve la géométrie : {composed.preserves_geometry()}")
print(f"  - Préserve la topologie : {composed.preserves_topology()}")
print(f"  - Invertible : {composed.invertible}")
print(f"  - Coût total : {composed.cost}")
print()

# Tester l'application
point = (1.0, 0.0)
print(f"Point initial : {point}")
print(f"Après translation : {translation.apply(point)}")
print(f"Après rotation : {rotation.apply(point)}")
print(f"Après composition complète : {composed.apply(point)}")
print()

# Tester l'inverse
print("Inversion de la composition :")
inverse_composed = composed.get_inverse()
if inverse_composed:
    print(f"  - Inverse existe : {inverse_composed.name}")
    transformed = composed.apply(point)
    inverted = inverse_composed.apply(transformed)
    print(f"  - Original : {point}")
    print(f"  - Après composition : {transformed}")
    print(f"  - Après inverse : {inverted}")
    print(f"  - Retrouvé l'original : {abs(inverted[0] - point[0]) < 1e-5 and abs(inverted[1] - point[1]) < 1e-5}")
print()

# ============================================================================
# PARTIE 4 : MacroTransformation
# ============================================================================
print("4. Macro-transformation (opération encapsulée)")
print("-" * 70)

# Créer une macro pour "place_on_table"
place_on_table_steps = [
    create_translation_transformation(0.0, 1.0),  # lever
    create_rotation_transformation(0.0, "reset_rotation"),  # rotation nulle
    create_translation_transformation(0.0, -1.0),  # descendre sur la table
]

# Créer la composition de base
place_on_table_composite = place_on_table_steps[0] @ place_on_table_steps[1] @ place_on_table_steps[2]

# Créer la macro
place_on_table = MacroTransformation(
    name="place_on_table",
    base_transformation=place_on_table_composite,
    description="Place un objet sur la table"
)

print(f"Macro : {place_on_table}")
print(f"  - Préserve la topologie : {place_on_table.preserves.preserves_topology}")
print(f"  - Préserve la géométrie : {place_on_table.preserves.preserves_geometry}")
print(f"  - Invertible : {place_on_table.invertible}")
print()

# ============================================================================
# PARTIE 5 : Topologie
# ============================================================================
print("5. Topologie et transformations topologiques")
print("-" * 70)

from core.topology import TopologySignature, DISK_TOPOLOGY, RING_TOPOLOGY, SEGMENT_TOPOLOGY
from core.transformations import CutTransformation, MergeTransformation

print(f"Disque : {DISK_TOPOLOGY}")
print(f"  - Est solide : {DISK_TOPOLOGY.is_solid()}")
print(f"  - Est simple : {DISK_TOPOLOGY.is_simple()}")
print()

print(f"Anneau : {RING_TOPOLOGY}")
print(f"  - Est solide : {RING_TOPOLOGY.is_solid()}")
print(f"  - A des trous : {RING_TOPOLOGY.holes > 0}")
print()

# Appliquer une découpe à un disque
cut = CutTransformation()
cut_topology = cut.apply(DISK_TOPOLOGY)
print(f"Disque -> après découpe : {cut_topology}")
print(f"  - Composantes connexes : {cut_topology.connected_components} (était 1)")
print()

# Appliquer une fusion
merge = MergeTransformation()
merged_topology = merge.apply(cut_topology)
print(f"Après fusion : {merged_topology}")
print(f"  - Composantes connexes : {merged_topology.connected_components}")
print()

# ============================================================================
# PARTIE 6 : TransformGraph
# ============================================================================
print("6. TransformGraph - Historique et suivi")
print("-" * 70)

from core.transform_graph import TransformGraph, TransformNode, create_linear_graph
from core.topology import DISK_TOPOLOGY

# Créer un graphe linéaire
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

print(f"Graphe créé : {graph}")
print()

# Afficher l'historique
summary = graph.get_history_summary()
print("Résumé de l'historique :")
for key, value in summary.items():
    print(f"  - {key} : {value}")
print()

# Afficher le chemin
path_names = graph.current.get_path_names()
print(f"Chemin des transformations : {' -> '.join(path_names)}")
print()

# Vérifier les invariants
current_invariants = graph.get_current_invariants()
print(f"Invariants préservés : {current_invariants}")
print(f"  - Préserve la topologie : {current_invariants.preserves_topology}")
print(f"  - Préserve la géométrie : {current_invariants.preserves_geometry}")
print()

# ============================================================================
# PARTIE 7 : Démonstration intégrée avec World
# ============================================================================
print("7. Intégration avec le système existant")
print("-" * 70)

from core.world import World, Entity
from core.transform import Transform2D

# Créer un monde simple
world = World()
world.add_entity(Entity(
    name="ball",
    kind="ball",
    properties={'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0, 'm': 1.0}
))

print(f"Monde créé avec entité : ball")
print()

# Créer un Transform2D qui est maintenant une transformation de premier ordre
T = Transform2D.translation(5.0, 0.0).compose(Transform2D.rotation(pi/2))
print(f"Transform2D : {T}")

# Appliquer à un point
p = (1.0, 0.0)
pt = T.transform_point(*p)
print(f"Point {p} -> {pt}")

# Vérifier l'inverse
T_inv = T.inverse()
ptt = T_inv.transform_point(*pt)
print(f"Inverse : {ptt} (original : {p})")
print(f"Précision : {abs(ptt[0] - p[0]) < 1e-10 and abs(ptt[1] - p[1]) < 1e-10}")
print()

# ============================================================================
# PARTIE 8 : Démonstration conceptuelle - Cercle vers Ellipse
# ============================================================================
print("8. Déformation : Cercle -> Ellipse")
print("-" * 70)

from core.deformation import StretchDeformation

# Créer une déformation d'étirement
stretch = StretchDeformation(stretch_x=2.0, stretch_y=3.0)
print(f"Déformation : {stretch.name}")
print(f"  - Préserve la topologie : {stretch.topology_preservation.preserves_connected_components}")
print(f"  - Préserve les trous : {stretch.topology_preservation.preserves_holes}")
print(f"  - Préserve les distances : False (stretch modifie la géométrie)")
print()

# Appliquer à un cercle conceptuel
print("Cercle (r=1) -> Ellipse (via stretch(2,3))")
print("  - Topologie initiale : DISK (1 composante, 0 trous)")
print("  - Topologie finale : DISK (inchangée)")
print("  - Géométrie modifiée : Oui")
print("  - Invariants préservés : connectivité, trous, dimension")
print()

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("=" * 70)
print("✓ PROPRIÉTÉS FONDAMENTALES DÉMONTRÉES")
print("=" * 70)
print()
print("1. Petit ensemble de transformations primitives")
print("   + composition")
print("   = grand nombre de comportements")
print()
print("2. Invariants explicites pour chaque transformation")
print("   → permet de raisonner sur ce qui change et ce qui reste")
print()
print("3. Topologie séparée de la géométrie")
print("   → permet de distinguer les propriétés fondamentales")
print()
print("4. TransformGraph pour suivre l'historique")
print("   → permet de répondre à : 'Quelles transformations ont produit cet état ?'")
print()
print("5. MacroTransformation pour encapsuler des compositions fréquentes")
print("   → permet de créer de nouvelles opérations sans modifier le code")
print()
print("=" * 70)
print("✓ DÉMO TRANSFORMATION ALGEBRA TERMINÉE")
print("=" * 70)
