import pygame
import math


class DebugObjectController:
    def __init__(self, move_speed=0.05):
        self.move_speed = move_speed
        self.rotation_speed_deg = 2.0
        self.targets = []
        self.current_index = 0
        self.m_was_pressed = False
        self.r_was_pressed = False

    def _build_target_list(self, world_objects):
        """
        Build list of controllable objects.
        Includes all world objects that have a transform.
        """
        self.targets = []

        for obj in world_objects:
            if hasattr(obj, "transform"):
                self.targets.append(obj)

        if self.current_index >= len(self.targets):
            self.current_index = 0

    def update(self, world_objects):
        keys = pygame.key.get_pressed()

        # Rebuild target list each frame (safe if objects change)
        self._build_target_list(world_objects)

        if not self.targets:
            return None

        # ---- Toggle target with M ----
        if keys[pygame.K_m] and not self.m_was_pressed:
            self.current_index = (self.current_index + 1) % len(self.targets)
            self.m_was_pressed = True

        elif not keys[pygame.K_m]:
            self.m_was_pressed = False

        target = self.targets[self.current_index]
        target_transform = getattr(target, "transform", None)

        # ---- Snap rotation to nearest 90° with R ----
        if keys[pygame.K_r] and not self.r_was_pressed:
            if target_transform is not None and hasattr(target_transform, "yaw"):
                current_deg = math.degrees(target_transform.yaw)
                snapped = round(current_deg / 90.0) * 90.0
                target_transform.yaw = math.radians(snapped)
            self.r_was_pressed = True

        elif not keys[pygame.K_r]:
            self.r_was_pressed = False

        # ---- Movement ----
        if target_transform is not None:
            if keys[pygame.K_UP]:
                target_transform.position[2] -= self.move_speed
            if keys[pygame.K_DOWN]:
                target_transform.position[2] += self.move_speed
            if keys[pygame.K_LEFT]:
                target_transform.position[0] -= self.move_speed
            if keys[pygame.K_RIGHT]:
                target_transform.position[0] += self.move_speed
            if keys[pygame.K_PAGEUP]:
                target_transform.position[1] += self.move_speed
            if keys[pygame.K_PAGEDOWN]:
                target_transform.position[1] -= self.move_speed

            # ---- Rotation (Hold TAB) ----
            if keys[pygame.K_TAB]:
                if hasattr(target_transform, "yaw"):
                    target_transform.yaw += math.radians(self.rotation_speed_deg)

        return target_transform
