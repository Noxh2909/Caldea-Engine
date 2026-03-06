import pygame


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