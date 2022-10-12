import numpy as np

FPS = 60


RES_canvas = (560,560)
RES_network = (256,560)
RES_control = (816,256)

SURF_control_offset = (0, RES_canvas[1])
SURF_network_offset = (RES_canvas[0], 0)

RES = (RES_canvas[0] + RES_network[0], RES_canvas[1] + RES_control[1])
# RES = (1000)


CL_BLACK = "#000000"
CL_GRAY20 = "#343434"
CL_GRAY10 = "#1c1c1c"
CL_GRAY5 = "#0d0d0d"

CL_RED = np.array([255, 0, 0])
CL_GREEN = np.array([0, 255, 0])


def make_gen_2d(i_count, j_count):
    def gen():
        for i in range(i_count):
            for j in range(j_count):
                yield (i, j)
    return gen


def make_gen_2d(i_start, i_end, j_start, j_end):
    def gen():
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                yield (i, j)
    return gen
