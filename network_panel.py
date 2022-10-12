import pygame

from constants import *

class Network_Panel:
    def __init__(self, surface):
        self.surface = surface

        self.font = pygame.font.SysFont("gost", 48)
        self.text = self.font.render(f"None", True, "#ffffff")

        class Progress_Bar:
            def __init__(self, surface, pos):
                self.surface = surface
                self.pos = pos
                self.height = 10
                self.length = 100
                self.progress = 0

            def update(self, progress):
                self.progress = progress


        self.bars = [Progress_Bar(self.surface, (10,(y*2)*10+10)) for y in range(10)]



    def update(self, value):
        self.text = self.font.render(f"{value}", True, "#ffffff")


    def draw(self):
        self.surface.fill(CL_GRAY10)
        self.surface.blit(self.text, (120,5))

        for bar in self.bars:
            pygame.draw.rect(self.surface, (CL_GREEN - CL_RED) * bar.progress + CL_RED,
                            (bar.pos[0], bar.pos[1], bar.length*bar.progress, bar.height))



if __name__ == "__main__":

    SURF_network_offset = (0,0)

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(RES_network)

    panel = Network_Panel(screen)
    panel.update(5)

    progresses = np.random.uniform(0, 1, (10))
    for i in range(10):
        panel.bars[i].update(progresses[i])

    while(1):
        panel.draw()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit() # Exit button
