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
    Adaptateur legacy : convertit une entité mur au format ancien
    en VerticalPlaneCollider + Material.
    
    Formats supportés:
    1. Avec side :
           orientation="vertical", x: position, side: "left"|"right", restitution: (optionnel)
       
    2. Sans side (compatibilité avec les tests existants) :
           orientation="vertical", x: position, restitution: (optionnel)
       Dans ce cas, la direction est déduite de la position :
       - x >= 0 → mur droit (normal_x=-1.0)
       - x < 0 → mur gauche (normal_x=+1.0)
    
    Retourne None si l'entité ne correspond pas au format legacy.
    """
    if entity_properties.get("orientation") != "vertical":
        return None
    
    wall_x = float(entity_properties.get("x", 0.0))
    side = entity_properties.get("side", "")
    restitution = float(entity_properties.get("restitution", DEFAULT_MATERIAL.restitution))
    material = Material(restitution=restitution)
    
    if side in ("left", "right"):
        # Format avec side explicite
        if side == "left":
            collider = VerticalPlaneCollider(x=wall_x, normal_x=+1.0)
        else:  # side == "right"
            collider = VerticalPlaneCollider(x=wall_x, normal_x=-1.0)
    else:
        # Format sans side : déduire la direction de la position x
        # x >= 0 → mur droit (bloque x > wall_x)
        # x < 0 → mur gauche (bloque x < wall_x)
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
    Collision avec des murs verticaux.
    
    Les murs peuvent être définis de deux manières (par ordre de priorité) :
    
    1. (Moderne) Comme entités avec VerticalPlaneCollider + Material dans leurs propriétés
       Exemple :
           Entity(
               name="mur_droit",
               kind="wall",
               properties={
                   "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                   "material": Material(restitution=0.0),
               }
           )
       
       Format legacy (avec side) aussi supporté via adaptateur automatique :
           Entity(
               name="mur_droit",
               kind="wall", 
               properties={"x": 5.0, "orientation": "vertical", "side": "right", "restitution": 0.0}
           )
    
    2. (Legacy) Via les paramètres min_x/max_x du constructeur pour rétrocompatibilité
    
    Applique une restitution sur vx en cas de collision.
    La collision ne modifie que x et vx.
    y, vy, et les relations de support restent inchangés.
    
    Ajoute une relation "touching" temporaire lors du contact.
    
    Architecture :
        - VerticalPlaneCollider : géométrie (x, normal_x)
        - Material : propriétés physiques (restitution)
        - _legacy_wall_to_collider_and_material : adaptateur pour l'ancien format
    """
    min_x: float = -float("inf")
    max_x: float = float("inf")
    restitution: float = 0.0

    def resolve(
        self,
        world,
        transformation: Transformation,
    ) -> Transformation:
        # D'abord, essayer de découvrir les (collider, material) depuis le World
        wall_components = self._get_wall_colliders_and_materials_from_world(world)
        
        if wall_components:
            # Utiliser les composants découverts dans le World (collider + material)
            # Cela inclut à la fois le format moderne (collider + material explicites)
            # et le format legacy (side, orientation, etc.) via l'adaptateur
            return self._resolve_with_colliders_and_materials(
                world, transformation, wall_components
            )
        else:
            # Rétrocompatibilité : utiliser min_x/max_x de l'instance
            return self._resolve_with_bounds(
                world, transformation, self.min_x, self.max_x, self.restitution
            )
    
    def _get_wall_colliders_and_materials_from_world(
        self, world
    ) -> list[tuple[str, VerticalPlaneCollider, Material]]:
        """
        Découvre les (collider, material) de mur verticaux depuis les entités du World.
        Retourne une liste de (entity_name, collider, material).
        
        Les colliders sont découverts dans l'ordre de priorité suivant :
        1. Entités avec "collider" PlaneCollider et optionnellement "material" (moderne)
        2. Entités avec "collider" VerticalPlaneCollider et optionnellement "material" (moderne)
        3. Entités legacy avec kind="wall", orientation="vertical", side, etc.
        
        Note : PlaneCollider est converti en VerticalPlaneCollider pour compatibilité
        avec la méthode de résolution existante.
        """
        wall_components: list[tuple[str, VerticalPlaneCollider, Material]] = []
        
        for entity in world.entities.values():
            # Chemin moderne : collider explicite dans les propriétés
            collider = entity.properties.get("collider")
            
            # Priorité 1 : PlaneCollider (géométrie moderne)
            if collider is not None and isinstance(collider, PlaneCollider):
                # Material peut être fourni séparément
                material = entity.properties.get("material")
                if material is None or not isinstance(material, Material):
                    material = DEFAULT_MATERIAL
                # Convertir PlaneCollider en VerticalPlaneCollider pour la résolution
                # Pour l'instant, nous ne supportons que les plans verticaux via PlaneCollider
                vpc = self._plane_collider_to_vertical_plane(collider, entity.name)
                if vpc is not None:
                    wall_components.append((entity.name, vpc, material))
                continue
            
            # Priorité 2 : VerticalPlaneCollider (géométrie moderne explicite)
            if collider is not None and isinstance(collider, VerticalPlaneCollider):
                # Material peut être fourni séparément
                material = entity.properties.get("material")
                if material is None or not isinstance(material, Material):
                    material = DEFAULT_MATERIAL
                wall_components.append((entity.name, collider, material))
                continue
            
            # Chemin legacy : conversion depuis le format side
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
            # Vérifier si la position proposée dépasse ou atteint les limites
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

                # Gérer le rebond sur vx (seulement si on se déplaçait vers le mur)
                vx_change = proposed_vx.get(entity_name)
                if vx_change is not None:
                    if vx_change.after > 0:
                        # Rebond vers la gauche
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
                    # vx n'a pas changé dans cette transformation, mais on a quand même une collision
                    # On doit inverser vx si la balle se déplaçait vers le mur
                    # On utilise la valeur actuelle de vx dans l'entité
                    entity = world.entities.get(entity_name)
                    if entity is not None and "vx" in entity.properties:
                        current_vx = float(entity.properties["vx"])
                        if current_vx > 0:
                            new_vx = -current_vx * restitution
                            # Normaliser -0.0 à 0.0
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

                # Gérer le rebond sur vx
                vx_change = proposed_vx.get(entity_name)
                if vx_change is not None:
                    if vx_change.after < 0:
                        # Rebond vers la droite
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
                    # vx n'a pas changé dans cette transformation
                    entity = world.entities.get(entity_name)
                    if entity is not None and "vx" in entity.properties:
                        current_vx = float(entity.properties["vx"])
                        if current_vx < 0:
                            new_vx = -current_vx * restitution
                            # Normaliser -0.0 à 0.0
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

        # Ajouter les nouveaux changements créés par la collision
        # (comme vx qui n'était pas dans la transformation originale)
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
        Résout les collisions en utilisant une liste de (collider, material).
        
        Pour chaque entité avec une position x proposée, vérifie chaque composant.
        Si une collision est détectée, corrige la position et la vitesse,
        et ajoute une relation "touching" avec le nom du mur.
        
        Le collider définit la géométrie (position, normale).
        Le material définit le comportement physique (restitution).
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
                    # Mur droit : bloque x > collider.x
                    # Collision si :
                    # 1. La balle traverse de gauche à droite (current_x <= collider.x et new_x > collider.x)
                    # 2. La balle est déjà à droite et se déplace vers la droite (current_x > collider.x et new_x > current_x)
                    # 3. La balle est déjà à droite (même si elle ne bouge pas, on la ramène)
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
                        
                        # Appliquer restitution seulement si la balle se déplaçait vers le mur
                        # (current_x <= collider.x et new_x > collider.x)
                        moving_toward_wall = (current_x <= collider.x)
                        self._apply_restitution(
                            world, replacements, proposed_vx, entity_name,
                            material.restitution, True  # moving right
                        ) if moving_toward_wall else None
                    
                elif collider.is_left_facing:
                    # Mur gauche : bloque x < collider.x
                    # Collision si :
                    # 1. La balle traverse de droite à gauche (current_x >= collider.x et new_x < collider.x)
                    # 2. La balle est déjà à gauche et se déplace vers la gauche (current_x < collider.x et new_x < current_x)
                    # 3. La balle est déjà à gauche (même si elle ne bouge pas, on la ramène)
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
                        
                        # Appliquer restitution seulement si la balle se déplaçait vers le mur
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
            # Vérifier si la relation existe déjà pour éviter les doublons
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
        Convertit un PlaneCollider en VerticalPlaneCollider si c'est un plan vertical.
        
        Un plan est vertical si sa normale dans le monde est approximativement (±1, 0).
        Retourne None si le plan n'est pas vertical.
        """
        # Calculer la normale dans le repère monde
        nx, ny = plane_collider.world_normal()
        
        # Vérifier si le plan est vertical (normale principalement horizontale)
        # Un plan vertical a |ny| ≈ 0 et |nx| ≈ 1
        if abs(ny) > 0.001:  # Pas exactement horizontal
            return None
        
        # Le plan est vertical. Déterminer la position x et la normale_x.
        # La normale monde nx devrait être proche de ±1.
        # Le signe de nx détermine la direction.
        
        # Trouver la position du plan le long de l'axe x.
        # Le plan passe par un point dans son repère local transformé.
        # Dans le repère local, le plan est à offset le long de la normale.
        # Mais pour un plan vertical simple, nous pouvons utiliser la position du transform.
        
        # Pour un PlaneCollider avec transform=translation(tx, ty), le plan est à :
        #   normal_local . (x_local, y_local) = offset
        # Dans le monde : normal_world . (x_world - origin_world, y_world - origin_world) = offset
        # 
        # Si le transform est translation(tx, ty) + rotation(theta) + ...
        # L'origine monde est à (tx, ty).
        # 
        # Pour une normale locale (nx_local, ny_local) = (1, 0) ou (-1, 0) :
        #   Dans le monde : nx_world = cos(theta) * nx_local - sin(theta) * ny_local
        #                 ny_world = sin(theta) * nx_local + cos(theta) * ny_local
        # 
        # Si le plan est vertical dans le monde, ny_world ≈ 0, donc :
        #   sin(theta) * nx_local + cos(theta) * ny_local ≈ 0
        # 
        # Pour une normale locale (1, 0) : sin(theta) ≈ 0, donc theta ≈ 0 ou pi
        # Pour une normale locale (-1, 0) : -sin(theta) ≈ 0, même condition
        # 
        # Si theta ≈ 0 : nx_world ≈ 1, ny_world ≈ 0
        # Si theta ≈ pi : nx_world ≈ -1, ny_world ≈ 0
        
        # La position x du plan dans le monde dépend du transform.
        # Pour un transform simple de translation : x = tx
        # Pour un transform avec rotation : il faut calculer où le plan se trouve.
        
        # Calculer un point sur le plan dans le repère local.
        # Le plan est à distance offset de l'origine locale, dans la direction de la normale.
        # Un point sur le plan en local : (offset * nx_local, offset * ny_local)
        # Dans le monde : ce point est transformé.
        
        transform = plane_collider.transform or Transform2D.identity()
        
        # Point sur le plan dans le repère local (si offset = 0, c'est l'origine locale)
        # Pour offset = 0, le plan passe par l'origine locale.
        # Donc dans le monde, le plan passe par transform(0, 0) = (tx, ty)
        # Mais attendez, ce n'est valable que si offset = 0.
        
        # Calculer la position du plan dans le monde.
        # Le plan est défini par : normal_local . (x_local, y_local) = offset
        # Dans le monde : normal_world . (p_world - origin_world) = offset
        # où origin_world = transform(0, 0)
        
        # Calculer l'origine dans le monde
        origin_world_x, origin_world_y = transform.transform_point(0, 0)
        
        # Le plan dans le monde : nx * (x - origin_world_x) + ny * (y - origin_world_y) = offset
        # Pour un plan vertical : ny ≈ 0, donc : nx * (x - origin_world_x) = offset
        # Donc : x = origin_world_x + offset / nx
        
        # Mais nx devrait être ±1 (la normale est unitaire), et offset est généralement 0.
        x_position = origin_world_x + plane_collider.offset * nx
        
        # Déterminer normal_x pour VerticalPlaneCollider
        # Si nx ≈ +1 : le côté interdit est x > x_position (mur gauche)
        # Si nx ≈ -1 : le côté interdit est x < x_position (mur droit)
        # 
        # Mais dans VerticalPlaneCollider :
        #   normal_x = +1 → bloque x < x (mur gauche)
        #   normal_x = -1 → bloque x > x (mur droit)
        # 
        # Donc :
        #   Si nx ≈ -1 : normal_x = -1 (mur droit)
        #   Si nx ≈ +1 : normal_x = +1 (mur gauche)
        
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
        """Applique la restitution à vx si l'entité se déplaçait vers le mur."""
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
