import pygame
import tkinter
import time


from constants import *
from canvas import Canvas
from control_panel import Control_Panel
from network import Network
from network_panel import Network_Panel


class App:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.font.SysFont('arial', 36)


        self.screen = pygame.display.set_mode(RES)          # main screen
        self.surf_canvas = pygame.Surface((RES_canvas))
        self.surf_network = pygame.Surface((RES_network))
        self.surf_control = pygame.Surface((RES_control))

        self.surf_canvas.fill(CL_BLACK)
        self.surf_network.fill(CL_GRAY10)
        self.surf_control.fill(CL_GRAY5)

        self.canvas = Canvas(self.surf_canvas)
        self.control_panel = Control_Panel(self.surf_control, self)
        self.network = Network(28*28, 10)
        self.network_panel = Network_Panel(self.surf_network)

        self.brush_size = 2
        self.click_code = 0

        self.update(self.brush_size)


    def run(self):
        while(1):
            # time_start = time.time()

            self.get_input()

            if self.click_code != 0:
                self.canvas.click_detect(self.brush_size, self.click_code)
                self.control_panel.click_detect()
                self.defineNumber()

            self.draw()


            # self.clock.tick(FPS)
            # print(time.time() - time_start)

    def reset(self):
        self.canvas.reset()

    def update(self, brush_size=None):
        if brush_size != None:
            self.control_panel.update(self.brush_size)



    def defineNumber(self):
        _in = self.canvas.get_data()
        self.network.run(_in)
        _out = self.network.get_choice()
        self.network_panel.update(_out)

        progresses = self.network.get_output()
        for i in range(10):
            self.network_panel.bars[i].update(progresses[0,i])




    def get_input(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                exit() # Exit button

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit() # ESC --> exit
                elif event.key == pygame.K_r:
                    self.reset() # R --> reset canvas
                # Change brush size
                elif event.key == pygame.K_KP_PLUS:
                    self.brush_size = self.brush_size + 1
                    self.update(brush_size=self.brush_size)
                elif event.key == pygame.K_KP_MINUS:
                    if self.brush_size <= 1:
                        return
                    self.brush_size = self.brush_size - 1
                    self.update(brush_size=self.brush_size)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.click_code = 1
                elif event.button == 3:
                    self.click_code = -1
            elif event.type == pygame.MOUSEBUTTONUP:
                self.click_code = 0

    def draw(self):
        self.canvas.draw()
        self.control_panel.draw()
        self.network_panel.draw()

        self.screen.blits(( (self.surf_canvas, (0,0)),
                            (self.surf_network, (RES_canvas[0],0)),
                            (self.surf_control, (0,RES_canvas[1]))
                            ))

        pygame.display.update()



if __name__ == "__main__":
    app = App()
    app.run()
