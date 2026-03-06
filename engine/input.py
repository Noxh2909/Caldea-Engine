import pygame


class InputState:
    """
    Collects and normalizes raw input into logical actions.

    - Movement & sprint: continuous (hold)
    - Jump: edge-triggered (press once)
    - Crouch:
        * crouch_hold -> HOLD for normal crouch
        * crouch_tap  -> TAP for slide
    """

    def __init__(self):
        """
        Docstring für __init__

        :param self: The object itself
        """
        self.actions = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "sprint": False,
            "crouch_hold": False,
            "crouch_tap": False,
            "jump": False,
            "toggle_third_person": False,
        }

        # previous key states for edge detection
        self._prev_jump = False
        self._prev_crouch = False
        self._prev_toggle_third_person = False

    def update(self):
        """
        Docstring für update

        :param self: The object itself
        """
        keys = pygame.key.get_pressed()

        # movement (continuous)
        self.actions["forward"] = keys[pygame.K_w]
        self.actions["backward"] = keys[pygame.K_s]
        self.actions["left"] = keys[pygame.K_a]
        self.actions["right"] = keys[pygame.K_d]
        self.actions["sprint"] = keys[pygame.K_LSHIFT]

        # crouch HOLD
        self.actions["crouch_hold"] = keys[pygame.K_c]

        # crouch TAP (edge-triggered)
        crouch_now = keys[pygame.K_c]
        self.actions["crouch_tap"] = crouch_now and not self._prev_crouch
        self._prev_crouch = crouch_now

        # jump TAP (edge-triggered)
        jump_now = keys[pygame.K_SPACE]
        self.actions["jump"] = jump_now and not self._prev_jump
        self._prev_jump = jump_now

        toggle_now = keys[pygame.K_v]
        self.actions["toggle_third_person"] = (
            toggle_now and not self._prev_toggle_third_person
        )
        self._prev_toggle_third_person = toggle_now

        return self.actions
