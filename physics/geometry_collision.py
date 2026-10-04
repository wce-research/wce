"""
Collisions géométriques générales pour WCE.

Ce module fournit des interactions de collision basées sur la géométrie
PlaneCollider + Transform2D.
"""

from dataclasses import dataclass
from typing import Optional

from core.transformation import (
    Change,
    RelationChange,
    Transformation,
    TransformationSource,
)
from core.invariants import Invariant, InvariantSet
from core.transformations import DeclarativeTransformationSpec, TransformationDomain
from core.transform import Transform2D
from core.world import World, Entity

from physics.colliders import (
    PlaneCollider,
    Material,
    DEFAULT_MATERIAL,
)


@dataclass
class PlaneCollision:
    """
    Collision avec des plans 2D arbitraires définis par PlaneCollider.
    
    Cette interaction détecte et résout les collisions entre des entités mobiles
    et des plans statiques définis dans le World via des entités avec un PlaneCollider.
    
    Un plan peut être :
        - Vertical (mur)
        - Horizontal (sol)
        - Incliné (plan à un angle)
    
    La résolution applique :
        - Correction de la position pour rester du côté autorisé du plan
        - Application de la restitution à la composante normale de la vitesse
    
    Note : Cette interaction ne gère pour l'instant que les plans statiques.
    Les plans mobiles ne sont pas supportés.
    """
    
    restitution: float = 0.0
    support_name: str = "surface"
    
    def resolve(
        self,
        world: World,
        transformation: Transformation,
    ) -> Transformation:
        """
        Résout les collisions avec tous les plans du World.
        
        Pour chaque entité avec un PlaneCollider dans le World, vérifie si
        d'autres entités traversent le plan et corrige leur position/vitesse.
        """
        # Découvrir tous les plans dans le World
        planes: list[tuple[str, Entity, PlaneCollider, Material]] = []
        
        for entity_name, entity in world.entities.items():
            collider = entity.properties.get("collider")
            if isinstance(collider, PlaneCollider):
                material = entity.properties.get("material")
                if material is None or not isinstance(material, Material):
                    material = DEFAULT_MATERIAL
                planes.append((entity_name, entity, collider, material))
        
        if not planes:
            return transformation
        
        changes = list(transformation.changes)
        relation_changes = list(transformation.relation_changes)
        
        # Extraire les positions et vitesses proposées
        proposed_positions = {}  # entity_name -> (x, y)
        proposed_velocities = {}  # entity_name -> (vx, vy)
        current_positions = {}   # entity_name -> (x, y) from world
        current_velocities = {}  # entity_name -> (vx, vy) from world
        
        # D'abord, collecter les informations actuelles du monde
        for entity_name, entity in world.entities.items():
            if "x" in entity.properties:
                current_positions[entity_name] = (
                    float(entity.properties["x"]),
                    float(entity.properties.get("y", 0.0)),
                )
            if "vx" in entity.properties:
                current_velocities[entity_name] = (
                    float(entity.properties["vx"]),
                    float(entity.properties.get("vy", 0.0)),
                )
        
        # Collecter les changements proposés
        for change in changes:
            if change.property == "x":
                if change.entity not in proposed_positions:
                    proposed_positions[change.entity] = (0.0, 0.0)
                proposed_positions[change.entity] = (
                    change.after,
                    proposed_positions[change.entity][1],
                )
            elif change.property == "y":
                if change.entity not in proposed_positions:
                    proposed_positions[change.entity] = (0.0, 0.0)
                proposed_positions[change.entity] = (
                    proposed_positions[change.entity][0],
                    change.after,
                )
            elif change.property == "vx":
                if change.entity not in proposed_velocities:
                    proposed_velocities[change.entity] = (0.0, 0.0)
                proposed_velocities[change.entity] = (
                    change.after,
                    proposed_velocities[change.entity][1],
                )
            elif change.property == "vy":
                if change.entity not in proposed_velocities:
                    proposed_velocities[change.entity] = (0.0, 0.0)
                proposed_velocities[change.entity] = (
                    proposed_velocities[change.entity][0],
                    change.after,
                )
        
        # Pour les entités qui n'ont pas de changements de position/vitesse,
        # utiliser les valeurs actuelles
        for entity_name in current_positions:
            if entity_name not in proposed_positions:
                proposed_positions[entity_name] = current_positions[entity_name]
            if entity_name not in proposed_velocities and entity_name in current_velocities:
                proposed_velocities[entity_name] = current_velocities[entity_name]
        
        replacements: dict[tuple[str, str], Change] = {}
        collision_occurred = False
        touching_pairs: set[tuple[str, str]] = set()
        
        # Pour chaque entité mobile et chaque plan
        for entity_name, (new_x, new_y) in proposed_positions.items():
            for plane_name, plane_entity, plane_collider, material in planes:
                if entity_name == plane_name:
                    continue  # Une entité ne peut pas collider avec elle-même
                
                # Calculer la distance signée
                signed_dist = plane_collider.signed_distance(new_x, new_y)
                
                # Obtenir la position actuelle
                current_x, current_y = current_positions.get(entity_name, (new_x, new_y))
                current_dist = plane_collider.signed_distance(current_x, current_y)
                
                # Si on traverse du côté autorisé vers le côté interdit
                if current_dist <= 0 and signed_dist > 0:
                    collision_occurred = True
                    touching_pairs.add((entity_name, plane_name))
                    
                    # Projeter sur le plan
                    world_normal = plane_collider.world_normal()
                    nx, ny = world_normal
                    
                    # Corriger la position : se placer sur le plan
                    # Le plan est défini par : nx * (x - x0) + ny * (y - y0) = offset
                    # où (x0, y0) est un point sur le plan dans le monde
                    # Pour simplifier, utiliser la distance signée
                    # La position corrigée est : (new_x, new_y) - signed_dist * (nx, ny)
                    # Mais signed_dist > 0, donc on soustrait
                    corrected_x = new_x - signed_dist * nx
                    corrected_y = new_y - signed_dist * ny
                    
                    replacements[(entity_name, "x")] = Change(
                        entity=entity_name,
                        property="x",
                        before=new_x,
                        after=corrected_x,
                    )
                    replacements[(entity_name, "y")] = Change(
                        entity=entity_name,
                        property="y",
                        before=new_y,
                        after=corrected_y,
                    )
                    
                    # Appliquer la restitution à la vitesse normale
                    # La composante normale de la vitesse
                    vx, vy = proposed_velocities.get(entity_name, (0.0, 0.0))
                    normal_velocity = vx * nx + vy * ny
                    
                    if normal_velocity > 0:  # Se déplace vers le côté interdit
                        # Composante tangentielle (inchangée par la collision)
                        tangent_x = vx - normal_velocity * nx
                        tangent_y = vy - normal_velocity * ny
                        
                        # Composante normale après collision (avec restitution)
                        normal_velocity_after = -normal_velocity * material.restitution
                        
                        vx_after = tangent_x + normal_velocity_after * nx
                        vy_after = tangent_y + normal_velocity_after * ny
                        
                        replacements[(entity_name, "vx")] = Change(
                            entity=entity_name,
                            property="vx",
                            before=vx,
                            after=vx_after,
                        )
                        replacements[(entity_name, "vy")] = Change(
                            entity=entity_name,
                            property="vy",
                            before=vy,
                            after=vy_after,
                        )
        
        # Construire la liste des changements résolus
        resolved = []
        for change in changes:
            key = (change.entity, change.property)
            resolved.append(replacements.get(key, change))
        
        # Ajouter les nouveaux changements (vx, vy qui n'étaient pas dans la transformation originale)
        for key, replacement_change in replacements.items():
            if key not in [(c.entity, c.property) for c in changes]:
                resolved.append(replacement_change)
        
        # Ajouter les relations "touching"
        for entity_name, plane_name in touching_pairs:
            relation_exists = any(
                r.subject == entity_name and r.predicate == "touching" and r.object == plane_name
                for r in world.relations
            )
            if not relation_exists:
                relation_changes.append(
                    RelationChange(
                        action="add",
                        subject=entity_name,
                        predicate="touching",
                        object=plane_name,
                    )
                )
        
        suffix = "+plane_collision" if collision_occurred else ""
        
        spec = transformation.spec
        source = transformation.source
        if collision_occurred:
            spec = DeclarativeTransformationSpec(
                name=type(self).__name__, domain=TransformationDomain.PHYSICS,
                parameters={"source_name": type(self).__name__},
                preserves=InvariantSet({Invariant.MASS, Invariant.ENTITY_IDENTITY}),
                changes={"position", "velocity", "relations"})
            source = TransformationSource.INTERACTION
        return Transformation(
            name=transformation.name + suffix,
            changes=resolved,
            relation_changes=relation_changes,
            spec=spec, source=source,
        )
