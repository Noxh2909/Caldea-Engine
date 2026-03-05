import pygame


# ============================================================
# Base UI Element
# ============================================================

class UIElement:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.hovered = False
        self.active = False

    def update(self, mouse_pos, mouse_pressed):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, surface, font):
        pass


# ============================================================
# Button
# ============================================================

class UIButton(UIElement):
    def __init__(self, rect, text, callback):
        super().__init__(rect)
        self.text = text
        self.callback = callback

    def update(self, mouse_pos, mouse_pressed):
        super().update(mouse_pos, mouse_pressed)
        if self.hovered and mouse_pressed:
            self.callback()

    def draw(self, surface, font):
        color = (90, 90, 90) if not self.hovered else (130, 130, 130)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (40, 40, 40), self.rect, 2)

        text_surf = font.render(self.text, True, (255, 255, 255))
        surface.blit(text_surf, (self.rect.x + 8, self.rect.y + 5))


# ============================================================
# Value Control (< >)
# ============================================================

class UIValueControl(UIElement):
    def __init__(self, rect, label, getter, setter, step=0.1):
        super().__init__(rect)
        self.label = label
        self.getter = getter
        self.setter = setter
        self.step = step

    def update(self, mouse_pos, mouse_pressed):
        super().update(mouse_pos, mouse_pressed)
        if not mouse_pressed:
            return

        left = pygame.Rect(self.rect.x, self.rect.y, 30, self.rect.height)
        right = pygame.Rect(self.rect.right - 30, self.rect.y, 30, self.rect.height)

        if left.collidepoint(mouse_pos):
            self.setter(self.getter() - self.step)

        if right.collidepoint(mouse_pos):
            self.setter(self.getter() + self.step)

    def draw(self, surface, font):
        pygame.draw.rect(surface, (60, 60, 60), self.rect)
        pygame.draw.rect(surface, (40, 40, 40), self.rect, 2)

        left = pygame.Rect(self.rect.x, self.rect.y, 30, self.rect.height)
        right = pygame.Rect(self.rect.right - 30, self.rect.y, 30, self.rect.height)

        pygame.draw.rect(surface, (90, 90, 90), left)
        pygame.draw.rect(surface, (90, 90, 90), right)

        value_text = f"{self.label}: {self.getter():.2f}"
        text_surf = font.render(value_text, True, (255, 255, 255))
        surface.blit(text_surf, (self.rect.x + 40, self.rect.y + 5))

        surface.blit(font.render("<", True, (255, 255, 255)), (left.x + 10, left.y + 5))
        surface.blit(font.render(">", True, (255, 255, 255)), (right.x + 10, right.y + 5))


# ============================================================
# Dropdown
# ============================================================

class UIDropdown(UIElement):
    def __init__(self, rect, options, setter):
        super().__init__(rect)
        self.options = options
        self.index = 0
        self.setter = setter
        self.open = False

    def update(self, mouse_pos, mouse_pressed):
        super().update(mouse_pos, mouse_pressed)

        if self.hovered and mouse_pressed:
            self.open = not self.open

        if self.open and mouse_pressed:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i + 1) * self.rect.height,
                    self.rect.width,
                    self.rect.height,
                )
                if option_rect.collidepoint(mouse_pos):
                    self.index = i
                    self.setter(option)
                    self.open = False

    def draw(self, surface, font):
        pygame.draw.rect(surface, (70, 70, 70), self.rect)
        pygame.draw.rect(surface, (40, 40, 40), self.rect, 2)

        text_surf = font.render(self.options[self.index], True, (255, 255, 255))
        surface.blit(text_surf, (self.rect.x + 8, self.rect.y + 5))

        if self.open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i + 1) * self.rect.height,
                    self.rect.width,
                    self.rect.height,
                )
                pygame.draw.rect(surface, (60, 60, 60), option_rect)
                pygame.draw.rect(surface, (40, 40, 40), option_rect, 1)
                option_text = font.render(option, True, (255, 255, 255))
                surface.blit(option_text, (option_rect.x + 8, option_rect.y + 5))


# ============================================================
# Panel
# ============================================================

class UIPanel(UIElement):
    def __init__(self, rect):
        super().__init__(rect)
        self.children = []

    def add(self, element):
        self.children.append(element)

    def update(self, mouse_pos, mouse_pressed):
        for child in self.children:
            child.update(mouse_pos, mouse_pressed)

    def draw(self, surface, font):
        pygame.draw.rect(surface, (30, 30, 30), self.rect)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 2)

        for child in self.children:
            child.draw(surface, font)


# ============================================================
# UI Manager
# ============================================================

class UIManager:
    def __init__(self):
        self.enabled = False
        self.panel = None
        self.font = pygame.font.SysFont("consolas", 16)

    def toggle(self):
        self.enabled = not self.enabled

    def set_panel(self, panel):
        self.panel = panel

    def update(self, mouse_pos, mouse_pressed):
        if not self.enabled or not self.panel:
            return
        self.panel.update(mouse_pos, mouse_pressed)

    def draw(self, surface):
        if not self.enabled or not self.panel:
            return
        self.panel.draw(surface, self.font)