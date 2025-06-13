import random
import pygame
from settings import WIDTH, HEIGHT

class Food:
    # class‐level image (must be set in main.py before you create any Food())
    img = None

    __slots__ = ('x','y','size')

    def __init__(self):
        self.x = random.uniform(0, WIDTH-250)
        self.y = random.uniform(0, HEIGHT)
        # assume Food.img is already set and is square (w×w)
        w = Food.img.get_width()
        # store half-width as collision radius
        self.size = w / 2

    def draw(self, surface):
        rect = Food.img.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(Food.img, rect)
