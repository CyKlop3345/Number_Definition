import pygame
import random
import math
import numpy as np

from constants import RES_canvas


class Canvas:
    def __init__(self, surface):
        self.surface = surface

        self.res = (self.height, self.width) = (28, 28)
        self.size = RES_canvas[0] / self.res[0]

        class Cell:
            def __init__(self, pos):
                self.pos = pos
                self.value = 0


        self.cells = [Cell((x,y)) for y in range(self.res[1]) for x in range(self.res[0])]



    def draw(self):
        for cell in self.cells:
            pygame.draw.rect(self.surface, (cell.value,cell.value,cell.value),
                    (cell.pos[0]*self.size, cell.pos[1]*self.size, self.size, self.size))


    def click_detect(self, brush_size, mode):
        mouse_pos = (int(pygame.mouse.get_pos()[0] // self.size), int(pygame.mouse.get_pos()[1] // self.size))
        if (mouse_pos[0] > self.res[0]) or (mouse_pos[1] > self.res[0]):
            return

        brush_size_sq = (brush_size**2)
        for x in range(mouse_pos[0]-brush_size, mouse_pos[0]+brush_size):
            for y in range(mouse_pos[1]-brush_size, mouse_pos[1]+brush_size):
                if (x >= self.res[0]) or (y >= self.res[1]):
                    continue

                cell = self.cells[y*self.res[0]+x]
                cell_dist_sq = (cell.pos[0]-mouse_pos[0])**2 + (cell.pos[1]-mouse_pos[1])**2
                if (cell_dist_sq < brush_size_sq):

                    if mode == 1:
                        # value = math.sqrt((brush_size_sq - cell_dist_sq) / brush_size_sq) * 255   # More smoothly
                        # value = (brush_size_sq - cell_dist_sq) / brush_size_sq * 255
                        value = ((brush_size_sq - cell_dist_sq) / brush_size_sq)**2 * 255           # More sharply
                        if value > cell.value:
                            cell.value = value
                    if mode == -1:
                        # value = math.sqrt(cell_dist_sq / brush_size_sq) * 255     # More smoothly
                        value = math.sqrt(cell_dist_sq / brush_size_sq) * 255
                        value = (cell_dist_sq / brush_size_sq)**2 * 255             # More sharply
                        if value < cell.value:
                            cell.value = value



    def reset(self):
        for cell in self.cells:
            cell.value = 0


    def get_data(self):
        data = np.array([cell.value for cell in self.cells])
        return data/255



if __name__ == "__main__":

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(RES_canvas)

    canvas = Canvas(screen)

    while(1):
        canvas.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit() # Exit button
        pygame.display.update()
