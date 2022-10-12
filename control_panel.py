import pygame

from constants import *


class Control_Panel:
    def __init__(self, surface, subject):
        self.surface = surface
        self.font = pygame.font.SysFont("gost", 48)
        self.font_small = pygame.font.SysFont("gost", 32)

        class Button:
            def __init__(self, poses, text, action, font):
                self.text = font.render(text, True, "#ffffff")
                self.poses = poses
                self.action = action

        class Text:
            def __init__(self, pos, text, font):
                self.font = font
                self.pos = pos
                self.text = self.font.render(text, True, "#ffffff")

            def update(self, text):
                self.text = self.font.render(text, True, "#ffffff")

        self.btn_reset = Button((10,10,160,50), "Reset", subject.reset, self.font)
        self.txt_br_size = Text((10, 70), f"Brush size: {None}", self.font_small)



    def update(self, brush_size=None):
        if brush_size != None:
            self.txt_br_size.update(f"Brush size: {brush_size}")

    def click_detect(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos = (mouse_pos[0] - SURF_control_offset[0], mouse_pos[1] - SURF_control_offset[1])

        for button in [self.btn_reset]:
            if button.poses[0] < mouse_pos[0] <= button.poses[2] and button.poses[1] < mouse_pos[1] <= button.poses[3]:
                button.action()



    def draw(self):
        self.surface.fill(CL_GRAY5)
        # btn_reset
        pygame.draw.rect(self.surface, CL_GRAY20, self.btn_reset.poses)
        self.surface.blit(self.btn_reset.text, (self.btn_reset.poses[0]+10, self.btn_reset.poses[1]+10))
        # txt_brush_size
        self.surface.blit(self.txt_br_size.text, self.txt_br_size.pos)



if __name__ == "__main__":

    class Subject:
        def __init__(self):
            pass
        def reset(self):
            print("Reset")

    SURF_control_offset = (0,0)


    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(RES_control)

    control = Control_Panel(screen, Subject())

    while(1):
        control.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit() # Exit button

            elif event.type == pygame.MOUSEBUTTONDOWN:
                control.click_detect()
        pygame.display.update()
