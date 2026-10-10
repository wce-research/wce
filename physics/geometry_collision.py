"""
General geometric collisions for WCE.

This module provides collision interactions based on the geometry
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
    Collision with arbitrary 2D planes defined by PlaneCollider.
    
    This interaction detects and resolves collisions between moving entities
    and static planes defined in the World through entities with a PlaneCollider.
    
    A plane can be:
        - Vertical (wall)
        - Horizontal (ground)
        - Inclined (plane at an angle)
    
    The resolution applies:
        - Correction of the position to stay on the allowed side of the plane
        - Application of the restitution to the normal component of the velocity
    
    Note: This interaction only handles static planes for now.
    Moving planes are not supported.
    """
    
    restitution: float = 0.0
    support_name: str = "surface"
    
    def resolve(
        self,
        world: World,
        transformation: Transformation,
    ) -> Transformation:
        """
        Resolves the collisions with all the planes of the World.
        
        For each entity with a PlaneCollider in the World, checks whether
        other entities cross the plane and corrects their position/velocity.
        """
        # Discover all the planes in the World
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
        
        # Extract the proposed positions and velocities
        proposed_positions = {}  # entity_name -> (x, y)
        proposed_velocities = {}  # entity_name -> (vx, vy)
        current_positions = {}   # entity_name -> (x, y) from world
        current_velocities = {}  # entity_name -> (vx, vy) from world
        
        # First, collect the current information from the world
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
        
        # Collect the proposed changes
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
        
        # For the entities that have no position/velocity changes,
        # use the current values
        for entity_name in current_positions:
            if entity_name not in proposed_positions:
                proposed_positions[entity_name] = current_positions[entity_name]
            if entity_name not in proposed_velocities and entity_name in current_velocities:
                proposed_velocities[entity_name] = current_velocities[entity_name]
        
        replacements: dict[tuple[str, str], Change] = {}
        collision_occurred = False
        touching_pairs: set[tuple[str, str]] = set()
        
        # For each moving entity and each plane
        for entity_name, (new_x, new_y) in proposed_positions.items():
            for plane_name, plane_entity, plane_collider, material in planes:
                if entity_name == plane_name:
                    continue  # An entity cannot collide with itself
                
                # Compute the signed distance
                signed_dist = plane_collider.signed_distance(new_x, new_y)
                
                # Get the current position
                current_x, current_y = current_positions.get(entity_name, (new_x, new_y))
                current_dist = plane_collider.signed_distance(current_x, current_y)
                
                # If we cross from the allowed side to the forbidden side
                if current_dist <= 0 and signed_dist > 0:
                    collision_occurred = True
                    touching_pairs.add((entity_name, plane_name))
                    
                    # Project onto the plane
                    world_normal = plane_collider.world_normal()
                    nx, ny = world_normal
                    
                    # Correct the position: place it on the plane
                    # The plane is defined by: nx * (x - x0) + ny * (y - y0) = offset
                    # where (x0, y0) is a point on the plane in the world
                    # To simplify, use the signed distance
                    # The corrected position is: (new_x, new_y) - signed_dist * (nx, ny)
                    # But signed_dist > 0, so we subtract
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
                    
                    # Apply the restitution to the normal velocity
                    # The normal component of the velocity
                    vx, vy = proposed_velocities.get(entity_name, (0.0, 0.0))
                    normal_velocity = vx * nx + vy * ny
                    
                    if normal_velocity > 0:  # Moving towards the forbidden side
                        # Tangential component (unchanged by the collision)
                        tangent_x = vx - normal_velocity * nx
                        tangent_y = vy - normal_velocity * ny
                        
                        # Normal component after collision (with restitution)
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
        
        # Build the list of resolved changes
        resolved = []
        for change in changes:
            key = (change.entity, change.property)
            resolved.append(replacements.get(key, change))
        
        # Add the new changes (vx, vy that were not in the original transformation)
        for key, replacement_change in replacements.items():
            if key not in [(c.entity, c.property) for c in changes]:
                resolved.append(replacement_change)
        
        # Add the "touching" relations
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
