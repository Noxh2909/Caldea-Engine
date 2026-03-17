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
        self.tab_was_pressed = False

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

        # ---- Snap rotation to nearest 90° with TAB ----
        if keys[pygame.K_TAB] and not self.tab_was_pressed:
            if target_transform is not None:

                if hasattr(target_transform, "yaw"):
                    current_deg = math.degrees(target_transform.yaw)
                    snapped = round(current_deg / 90.0) * 90.0
                    target_transform.yaw = math.radians(snapped)

                if hasattr(target_transform, "roll"):
                    current_deg = math.degrees(target_transform.roll)
                    snapped = round(current_deg / 90.0) * 90.0
                    target_transform.roll = math.radians(snapped)
                    
                if hasattr(target_transform, "pitch"):
                    current_deg = math.degrees(target_transform.pitch)
                    snapped = round(current_deg / 90.0) * 90.0
                    target_transform.pitch = math.radians(snapped)

            self.tab_was_pressed = True

        elif not keys[pygame.K_TAB]:
            self.tab_was_pressed = False

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

            # ---- Continuous rotation ----
            if target_transform is not None:

                # Rotate yaw with key 1
                if keys[pygame.K_1] and hasattr(target_transform, "yaw"):
                    target_transform.yaw += math.radians(self.rotation_speed_deg)

                # Rotate roll with key 2
                if keys[pygame.K_2] and hasattr(target_transform, "roll"):
                    target_transform.roll += math.radians(self.rotation_speed_deg)
                    
                if keys[pygame.K_3] and hasattr(target_transform, "pitch"):
                    target_transform.pitch += math.radians(self.rotation_speed_deg)

        return target_transform
