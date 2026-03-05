import pygame


# ============================================================
# UI Value Control
# ============================================================

class UIValueControl:
    """
    UIValueControl
    =================

    Simple interactive value control element.

    Provides:
    - Label display
    - < and > button interaction
    - Getter / Setter binding

    Designed to integrate into the in-game debug overlay.
    """

    def __init__(self, x, y, width, label, getter, setter, step=0.1):
        """
        :param x: X screen position
        :param y: Y screen position
        :param width: Total control width
        :param label: Display label
        :param getter: Callable returning current value
        :param setter: Callable accepting new value
        :param step: Increment / decrement amount
        """
        self.rect = pygame.Rect(x, y, width, 30)
        self.label = label
        self.getter = getter
        self.setter = setter
        self.step = step

    # ============================================================
    # Update Logic
    # ============================================================

    def update(self, mouse_pos, mouse_pressed, keys):
        """
        Handle mouse interaction for increment / decrement buttons.
        """
        if not mouse_pressed:
            return

        left = pygame.Rect(self.rect.x, self.rect.y, 30, 30)
        right = pygame.Rect(self.rect.right - 30, self.rect.y, 30, 30)

        if left.collidepoint(mouse_pos):
            self.setter(self.getter() - self.step)

        if right.collidepoint(mouse_pos):
            self.setter(self.getter() + self.step)

    # ============================================================
    # Display Text
    # ============================================================

    def get_text(self):
        """
        Return formatted display string.
        """
        return f"{self.label}: {self.getter():.2f}"


# ============================================================
# UI Manager
# ============================================================

class UIManager:
    """
    UIManager
    =================

    Manages debug UI elements.

    Responsibilities:
    - Toggle UI visibility
    - Update registered UI controls
    - Provide formatted debug lines for HUD rendering
    """

    def __init__(self):
        self.elements = []
        self.enabled = False

    # ============================================================
    # State Control
    # ============================================================

    def toggle(self):
        """Toggle debug UI visibility."""
        self.enabled = not self.enabled

    def add(self, element):
        """Register a UI element."""
        self.elements.append(element)

    # ============================================================
    # Update Loop
    # ============================================================

    def update(self, mouse_pos, mouse_pressed, keys):
        """
        Update all registered UI elements.
        """
        if not self.enabled:
            return

        for element in self.elements:
            element.update(mouse_pos, mouse_pressed, keys)

    # ============================================================
    # Debug Text Output
    # ============================================================

    def get_lines(self):
        """
        Return formatted text lines for debug HUD rendering.
        """
        if not self.enabled:
            return []

        lines = []
        for element in self.elements:
            if hasattr(element, "get_text"):
                lines.append(element.get_text())

        return lines