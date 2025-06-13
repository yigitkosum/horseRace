import pygame

class Button:
    __slots__ = ('rect','text_surf','text_rect','callback','bg_color')

    def __init__(self, rect, text, font, callback,
                 bg_color=(200,200,200), text_color=(0,0,0)):
        self.rect      = pygame.Rect(rect)
        self.callback  = callback
        self.bg_color  = bg_color
        self.text_surf = font.render(text, True, text_color)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        surface.blit(self.text_surf, self.text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
