from dataclasses import dataclass

from core.transformation import (
    Change,
    RelationChange,
    Transformation,
    TransformationSource,
)
from core.invariants import Invariant, InvariantSet
from core.transformations import DeclarativeTransformationSpec, TransformationDomain


def _interaction_metadata(owner, transformation, occurred):
    if not occurred:
        return transformation.spec, transformation.source
    spec = DeclarativeTransformationSpec(
        name=type(owner).__name__, domain=TransformationDomain.PHYSICS,
        parameters={"source_name": type(owner).__name__},
        preserves=InvariantSet({Invariant.MASS, Invariant.ENTITY_IDENTITY}),
        changes={"position", "velocity", "relations"},
    )
    return spec, TransformationSource.INTERACTION
from core.transform import Transform2D

from physics.colliders import (
    DEFAULT_MATERIAL,
    Material,
    PlaneCollider,
    VerticalPlaneCollider,
    vertical_plane_to_plane_collider,
)


def _legacy_wall_to_collider_and_material(
    entity_properties: dict
) -> tuple[VerticalPlaneCollider, Material] | None:
    """
    Legacy adapter: converts a wall entity in the old format
    into VerticalPlaneCollider + Material.
    
    Supported formats:
    1. With side:
           orientation="vertical", x: position, side: "left"|"right", restitution: (optional)
       
    2. Without side (compatibility with the existing tests):
           orientation="vertical", x: position, restitution: (optional)
       In this case, the direction is deduced from the position:
       - x >= 0 → right wall (normal_x=-1.0)
       - x < 0 → left wall (normal_x=+1.0)
    
    Returns None if the entity does not match the legacy format.
    """
    if entity_properties.get("orientation") != "vertical":
        return None
    
    wall_x = float(entity_properties.get("x", 0.0))
    side = entity_properties.get("side", "")
    restitution = float(entity_properties.get("restitution", DEFAULT_MATERIAL.restitution))
    material = Material(restitution=restitution)
    
    if side in ("left", "right"):
        # Format with explicit side
        if side == "left":
            collider = VerticalPlaneCollider(x=wall_x, normal_x=+1.0)
        else:  # side == "right"
            collider = VerticalPlaneCollider(x=wall_x, normal_x=-1.0)
    else:
        # Format without side: deduce the direction from the x position
        # x >= 0 → right wall (blocks x > wall_x)
        # x < 0 → left wall (blocks x < wall_x)
        if wall_x >= 0:
            collider = VerticalPlaneCollider(x=wall_x, normal_x=-1.0)
        else:
            collider = VerticalPlaneCollider(x=wall_x, normal_x=+1.0)
    
    return (collider, material)


@dataclass
class GroundCollision:
    ground_y: float = 0.0
    restitution: float = 0.0
    support_name: str = "sol"

    def resolve(
        self,
        world,
        transformation: Transformation,
    ) -> Transformation:

        changes = list(transformation.changes)
        relation_changes = list(
            transformation.relation_changes
        )

        proposed_y = {}
        proposed_vy = {}

        for change in changes:
            if change.property == "y":
                proposed_y[change.entity] = change

            elif change.property == "vy":
                proposed_vy[change.entity] = change

        replacements = {}
        collision = False

        for entity_name, y_change in proposed_y.items():
            if y_change.after >= self.ground_y:
                continue

            collision = True

            replacements[
                (entity_name, "y")
            ] = Change(
                entity=entity_name,
                property="y",
                before=y_change.before,
                after=self.ground_y,
            )

            vy_change = proposed_vy.get(entity_name)

            if vy_change is not None and vy_change.after < 0:
                new_vy = (
                    -vy_change.after
                    * self.restitution
                )

                replacements[
                    (entity_name, "vy")
                ] = Change(
                    entity=entity_name,
                    property="vy",
                    before=vy_change.before,
                    after=new_vy,
                )

                if self.restitution == 0.0:
                    relation_changes.append(
                        RelationChange(
                            action="add",
                            subject=entity_name,
                            predicate="supported_by",
                            object=self.support_name,
                        )
                    )

        resolved = []

        for change in changes:
            key = (
                change.entity,
                change.property,
            )

            resolved.append(
                replacements.get(key, change)
            )

        suffix = (
            "+ground_collision"
            if collision
            else ""
        )

        spec, source = _interaction_metadata(self, transformation, collision)
        return Transformation(
            name=transformation.name + suffix,
            changes=resolved,
            relation_changes=relation_changes,
            spec=spec, source=source,
        )


@dataclass
class WallCollision:
    """
    Collision with vertical walls.
    
    Walls can be defined in two ways (in order of priority):
    
    1. (Modern) As entities with VerticalPlaneCollider + Material in their properties
       Example:
           Entity(
               name="mur_droit",
               kind="wall",
               properties={
                   "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                   "material": Material(restitution=0.0),
               }
           )
       
       Legacy format (with side) also supported through an automatic adapter:
           Entity(
               name="mur_droit",
               kind="wall", 
               properties={"x": 5.0, "orientation": "vertical", "side": "right", "restitution": 0.0}
           )
    
    2. (Legacy) Through the min_x/max_x parameters of the constructor for backward compatibility
    
    Applies a restitution on vx in case of collision.
    The collision only modifies x and vx.
    y, vy, and the support relations remain unchanged.
    
    Adds a temporary "touching" relation upon contact.
    
    Architecture:
        - VerticalPlaneCollider: geometry (x, normal_x)
        - Material: physical properties (restitution)
        - _legacy_wall_to_collider_and_material: adapter for the old format
    """
    min_x: float = -float("inf")
    max_x: float = float("inf")
    restitution: float = 0.0

    def resolve(
        self,
        world,
        transformation: Transformation,
    ) -> Transformation:
        # First, try to discover the (collider, material) from the World
        wall_components = self._get_wall_colliders_and_materials_from_world(world)
        
        if wall_components:
            # Use the components discovered in the World (collider + material)
            # This includes both the modern format (explicit collider + material)
            # and the legacy format (side, orientation, etc.) through the adapter
            return self._resolve_with_colliders_and_materials(
                world, transformation, wall_components
            )
        else:
            # Backward compatibility: use min_x/max_x of the instance
            return self._resolve_with_bounds(
                world, transformation, self.min_x, self.max_x, self.restitution
            )
    
    def _get_wall_colliders_and_materials_from_world(
        self, world
    ) -> list[tuple[str, VerticalPlaneCollider, Material]]:
        """
        Discovers the vertical wall (collider, material) from the entities of the World.
        Returns a list of (entity_name, collider, material).
        
        The colliders are discovered in the following order of priority:
        1. Entities with a PlaneCollider "collider" and optionally a "material" (modern)
        2. Entities with a VerticalPlaneCollider "collider" and optionally a "material" (modern)
        3. Legacy entities with kind="wall", orientation="vertical", side, etc.
        
        Note: PlaneCollider is converted into VerticalPlaneCollider for compatibility
        with the existing resolution method.
        """
        wall_components: list[tuple[str, VerticalPlaneCollider, Material]] = []
        
        for entity in world.entities.values():
            # Modern path: explicit collider in the properties
            collider = entity.properties.get("collider")
            
            # Priority 1: PlaneCollider (modern geometry)
            if collider is not None and isinstance(collider, PlaneCollider):
                # Material can be provided separately
                material = entity.properties.get("material")
                if material is None or not isinstance(material, Material):
                    material = DEFAULT_MATERIAL
                # Convert PlaneCollider into VerticalPlaneCollider for the resolution
                # For now, we only support vertical planes through PlaneCollider
                vpc = self._plane_collider_to_vertical_plane(collider, entity.name)
                if vpc is not None:
                    wall_components.append((entity.name, vpc, material))
                continue
            
            # Priority 2: VerticalPlaneCollider (explicit modern geometry)
            if collider is not None and isinstance(collider, VerticalPlaneCollider):
                # Material can be provided separately
                material = entity.properties.get("material")
                if material is None or not isinstance(material, Material):
                    material = DEFAULT_MATERIAL
                wall_components.append((entity.name, collider, material))
                continue
            
            # Legacy path: conversion from the side format
            if entity.kind == "wall" and entity.properties.get("orientation") == "vertical":
                legacy_result = _legacy_wall_to_collider_and_material(entity.properties)
                if legacy_result is not None:
                    collider, material = legacy_result
                    wall_components.append((entity.name, collider, material))
        
        return wall_components
    
    
    def _resolve_with_bounds(
        self,
        world,
        transformation: Transformation,
        min_x: float,
        max_x: float,
        restitution: float,
    ) -> Transformation:

        changes = list(transformation.changes)
        relation_changes = list(
            transformation.relation_changes
        )

        proposed_x = {}
        proposed_vx = {}

        for change in changes:
            if change.property == "x":
                proposed_x[change.entity] = change

            elif change.property == "vx":
                proposed_vx[change.entity] = change

        replacements = {}
        collision = False

        for entity_name, x_change in proposed_x.items():
            # Check whether the proposed position exceeds or reaches the limits
            if x_change.after >= max_x:
                collision = True

                replacements[
                    (entity_name, "x")
                ] = Change(
                    entity=entity_name,
                    property="x",
                    before=x_change.before,
                    after=max_x,
                )

                # Handle the bounce on vx (only if we were moving towards the wall)
                vx_change = proposed_vx.get(entity_name)
                if vx_change is not None:
                    if vx_change.after > 0:
                        # Bounce to the left
                        new_vx = -vx_change.after * restitution
                        replacements[
                            (entity_name, "vx")
                        ] = Change(
                            entity=entity_name,
                            property="vx",
                            before=vx_change.before,
                            after=new_vx,
                        )
                else:
                    # vx did not change in this transformation, but we still have a collision
                    # We must invert vx if the ball was moving towards the wall
                    # We use the current value of vx in the entity
                    entity = world.entities.get(entity_name)
                    if entity is not None and "vx" in entity.properties:
                        current_vx = float(entity.properties["vx"])
                        if current_vx > 0:
                            new_vx = -current_vx * restitution
                            # Normalise -0.0 to 0.0
                            if new_vx == 0.0:
                                new_vx = 0.0
                            replacements[
                                (entity_name, "vx")
                            ] = Change(
                                entity=entity_name,
                                property="vx",
                                before=current_vx,
                                after=new_vx,
                            )

            elif x_change.after <= min_x:
                collision = True

                replacements[
                    (entity_name, "x")
                ] = Change(
                    entity=entity_name,
                    property="x",
                    before=x_change.before,
                    after=min_x,
                )

                # Handle the bounce on vx
                vx_change = proposed_vx.get(entity_name)
                if vx_change is not None:
                    if vx_change.after < 0:
                        # Bounce to the right
                        new_vx = -vx_change.after * restitution
                        replacements[
                            (entity_name, "vx")
                        ] = Change(
                            entity=entity_name,
                            property="vx",
                            before=vx_change.before,
                            after=new_vx,
                        )
                else:
                    # vx did not change in this transformation
                    entity = world.entities.get(entity_name)
                    if entity is not None and "vx" in entity.properties:
                        current_vx = float(entity.properties["vx"])
                        if current_vx < 0:
                            new_vx = -current_vx * restitution
                            # Normalise -0.0 to 0.0
                            if new_vx == 0.0:
                                new_vx = 0.0
                            replacements[
                                (entity_name, "vx")
                            ] = Change(
                                entity=entity_name,
                                property="vx",
                                before=current_vx,
                                after=new_vx,
                            )

        resolved = []

        for change in changes:
            key = (
                change.entity,
                change.property,
            )

            resolved.append(
                replacements.get(key, change)
            )

        # Add the new changes created by the collision
        # (such as vx which was not in the original transformation)
        for key, replacement_change in replacements.items():
            if key not in [(c.entity, c.property) for c in changes]:
                resolved.append(replacement_change)

        suffix = (
            "+wall_collision"
            if collision
            else ""
        )

        spec, source = _interaction_metadata(self, transformation, collision)
        return Transformation(
            name=transformation.name + suffix,
            changes=resolved,
            relation_changes=relation_changes,
            spec=spec, source=source,
        )
    
    def _resolve_with_colliders_and_materials(
        self,
        world,
        transformation: Transformation,
        wall_components: list[tuple[str, VerticalPlaneCollider, Material]],
    ) -> Transformation:
        """
        Resolves the collisions using a list of (collider, material).
        
        For each entity with a proposed x position, checks each component.
        If a collision is detected, corrects the position and the velocity,
        and adds a "touching" relation with the name of the wall.
        
        The collider defines the geometry (position, normal).
        The material defines the physical behaviour (restitution).
        """
        changes = list(transformation.changes)
        relation_changes = list(transformation.relation_changes)
        
        proposed_x = {}
        proposed_vx = {}
        
        for change in changes:
            if change.property == "x":
                proposed_x[change.entity] = change
            elif change.property == "vx":
                proposed_vx[change.entity] = change
        
        replacements = {}
        collision_occurred = False
        touching_pairs: set[tuple[str, str]] = set()
        
        for entity_name, x_change in proposed_x.items():
            entity = world.entities.get(entity_name)
            if entity is None:
                continue
            
            current_x = float(entity.properties.get("x", x_change.before))
            new_x = x_change.after
            
            for wall_name, collider, material in wall_components:
                if collider.is_right_facing:
                    # Right wall: blocks x > collider.x
                    # Collision if:
                    # 1. The ball crosses from left to right (current_x <= collider.x and new_x > collider.x)
                    # 2. The ball is already on the right and is moving to the right (current_x > collider.x and new_x > current_x)
                    # 3. The ball is already on the right (even if it does not move, we bring it back)
                    if (new_x > collider.x and current_x <= collider.x) or \
                       (current_x > collider.x):
                        collision_occurred = True
                        touching_pairs.add((entity_name, wall_name))
                        
                        replacements[(entity_name, "x")] = Change(
                            entity=entity_name,
                            property="x",
                            before=x_change.before,
                            after=collider.x,
                        )
                        
                        # Apply restitution only if the ball was moving towards the wall
                        # (current_x <= collider.x and new_x > collider.x)
                        moving_toward_wall = (current_x <= collider.x)
                        self._apply_restitution(
                            world, replacements, proposed_vx, entity_name,
                            material.restitution, True  # moving right
                        ) if moving_toward_wall else None
                    
                elif collider.is_left_facing:
                    # Left wall: blocks x < collider.x
                    # Collision if:
                    # 1. The ball crosses from right to left (current_x >= collider.x and new_x < collider.x)
                    # 2. The ball is already on the left and is moving to the left (current_x < collider.x and new_x < current_x)
                    # 3. The ball is already on the left (even if it does not move, we bring it back)
                    if (new_x < collider.x and current_x >= collider.x) or \
                       (current_x < collider.x):
                        collision_occurred = True
                        touching_pairs.add((entity_name, wall_name))
                        
                        replacements[(entity_name, "x")] = Change(
                            entity=entity_name,
                            property="x",
                            before=x_change.before,
                            after=collider.x,
                        )
                        
                        # Apply restitution only if the ball was moving towards the wall
                        moving_toward_wall = (current_x >= collider.x)
                        self._apply_restitution(
                            world, replacements, proposed_vx, entity_name,
                            material.restitution, False  # moving left
                        ) if moving_toward_wall else None
        
        resolved = []
        
        for change in changes:
            key = (change.entity, change.property)
            resolved.append(replacements.get(key, change))
        
        for key, replacement_change in replacements.items():
            if key not in [(c.entity, c.property) for c in changes]:
                resolved.append(replacement_change)
        
        for entity_name, wall_name in touching_pairs:
            # Check whether the relation already exists to avoid duplicates
            relation_exists = any(
                r.subject == entity_name and r.predicate == "touching" and r.object == wall_name
                for r in world.relations
            )
            if not relation_exists:
                relation_changes.append(
                    RelationChange(
                        action="add",
                        subject=entity_name,
                        predicate="touching",
                        object=wall_name,
                    )
                )
        
        suffix = "+wall_collision" if collision_occurred else ""
        
        spec, source = _interaction_metadata(self, transformation, collision_occurred)
        return Transformation(
            name=transformation.name + suffix,
            changes=resolved,
            relation_changes=relation_changes,
            spec=spec, source=source,
        )
    
    def _plane_collider_to_vertical_plane(
        self, plane_collider: PlaneCollider, entity_name: str
    ) -> VerticalPlaneCollider | None:
        """
        Converts a PlaneCollider into a VerticalPlaneCollider if it is a vertical plane.
        
        A plane is vertical if its normal in the world is approximately (±1, 0).
        Returns None if the plane is not vertical.
        """
        # Compute the normal in the world frame
        nx, ny = plane_collider.world_normal()
        
        # Check whether the plane is vertical (mostly horizontal normal)
        # A vertical plane has |ny| ≈ 0 and |nx| ≈ 1
        if abs(ny) > 0.001:  # Not exactly horizontal
            return None
        
        # The plane is vertical. Determine the x position and the normal_x.
        # The world normal nx should be close to ±1.
        # The sign of nx determines the direction.
        
        # Find the position of the plane along the x axis.
        # The plane passes through a point in its transformed local frame.
        # In the local frame, the plane is at offset along the normal.
        # But for a simple vertical plane, we can use the position of the transform.
        
        # For a PlaneCollider with transform=translation(tx, ty), the plane is at:
        #   normal_local . (x_local, y_local) = offset
        # In the world: normal_world . (x_world - origin_world, y_world - origin_world) = offset
        # 
        # If the transform is translation(tx, ty) + rotation(theta) + ...
        # The world origin is at (tx, ty).
        # 
        # For a local normal (nx_local, ny_local) = (1, 0) or (-1, 0):
        #   In the world: nx_world = cos(theta) * nx_local - sin(theta) * ny_local
        #                 ny_world = sin(theta) * nx_local + cos(theta) * ny_local
        # 
        # If the plane is vertical in the world, ny_world ≈ 0, so:
        #   sin(theta) * nx_local + cos(theta) * ny_local ≈ 0
        # 
        # For a local normal (1, 0): sin(theta) ≈ 0, so theta ≈ 0 or pi
        # For a local normal (-1, 0): -sin(theta) ≈ 0, same condition
        # 
        # If theta ≈ 0: nx_world ≈ 1, ny_world ≈ 0
        # If theta ≈ pi: nx_world ≈ -1, ny_world ≈ 0
        
        # The x position of the plane in the world depends on the transform.
        # For a simple translation transform: x = tx
        # For a transform with rotation: we must compute where the plane lies.
        
        # Compute a point on the plane in the local frame.
        # The plane is at distance offset from the local origin, in the direction of the normal.
        # A point on the plane in local coordinates: (offset * nx_local, offset * ny_local)
        # In the world: this point is transformed.
        
        transform = plane_collider.transform or Transform2D.identity()
        
        # Point on the plane in the local frame (if offset = 0, it is the local origin)
        # For offset = 0, the plane passes through the local origin.
        # So in the world, the plane passes through transform(0, 0) = (tx, ty)
        # But wait, this only holds if offset = 0.
        
        # Compute the position of the plane in the world.
        # The plane is defined by: normal_local . (x_local, y_local) = offset
        # In the world: normal_world . (p_world - origin_world) = offset
        # where origin_world = transform(0, 0)
        
        # Compute the origin in the world
        origin_world_x, origin_world_y = transform.transform_point(0, 0)
        
        # The plane in the world: nx * (x - origin_world_x) + ny * (y - origin_world_y) = offset
        # For a vertical plane: ny ≈ 0, so: nx * (x - origin_world_x) = offset
        # So: x = origin_world_x + offset / nx
        
        # But nx should be ±1 (the normal is a unit vector), and offset is generally 0.
        x_position = origin_world_x + plane_collider.offset * nx
        
        # Determine normal_x for VerticalPlaneCollider
        # If nx ≈ +1: the forbidden side is x > x_position (left wall)
        # If nx ≈ -1: the forbidden side is x < x_position (right wall)
        # 
        # But in VerticalPlaneCollider:
        #   normal_x = +1 → blocks x < x (left wall)
        #   normal_x = -1 → blocks x > x (right wall)
        # 
        # So:
        #   If nx ≈ -1: normal_x = -1 (right wall)
        #   If nx ≈ +1: normal_x = +1 (left wall)
        
        normal_x = -1.0 if nx > 0 else 1.0
        
        return VerticalPlaneCollider(x=x_position, normal_x=normal_x)
    
    def _apply_restitution(
        self,
        world,
        replacements: dict,
        proposed_vx: dict,
        entity_name: str,
        restitution: float,
        is_moving_right: bool,
    ) -> None:
        """Applies the restitution to vx if the entity was moving towards the wall."""
        vx_change = proposed_vx.get(entity_name)
        if vx_change is not None:
            if (vx_change.after > 0 and is_moving_right) or \
               (vx_change.after < 0 and not is_moving_right):
                new_vx = -vx_change.after * restitution
                if new_vx == 0.0:
                    new_vx = 0.0
                replacements[(entity_name, "vx")] = Change(
                    entity=entity_name,
                    property="vx",
                    before=vx_change.before,
                    after=new_vx,
                )
        else:
            entity = world.entities.get(entity_name)
            if entity is not None and "vx" in entity.properties:
                current_vx = float(entity.properties["vx"])
                if (current_vx > 0 and is_moving_right) or \
                   (current_vx < 0 and not is_moving_right):
                    new_vx = -current_vx * restitution
                    if new_vx == 0.0:
                        new_vx = 0.0
                    replacements[(entity_name, "vx")] = Change(
                        entity=entity_name,
                        property="vx",
                        before=current_vx,
                        after=new_vx,
                    )
