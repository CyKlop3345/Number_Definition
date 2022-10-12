from constants import *

import numpy as np
import matplotlib.pyplot as plt
import random
import pygame
from pathlib import Path



# Mistakes calculation
def linear_loss(y, y_right):
    return np.sum(np.absolute(y-y_right))

def quadratic_loss(y, y_right):
    return np.sum(np.power(y-y_right, 2))

def cross_entropy_loss(y, y_right):
    return -np.sum(y_right * np.log(y))

def softmax(x):
    # Activation func for output layers
    # Converts a numeric array to фт array of probabilities
    out = np.exp(x)
    # out = 1 + x + 0.5*np.power(x, 2)
    sum = np.sum(out)
    if sum <= 1:
        return out
    return out / sum



# Activation funcs and derivatives
def sigmoid(x):
    ''' Hint
    x == 0    => y = 0.5
    x -> inf  => y -> 1
    x -> -inf => y -> 0
    '''
    out = np.copy(x)
    out = 1 / ( 1 + np.exp(-out) )
    return out

def sigmoid_deriv(x):
    out = np.copy(x)
    out = sigmoid(out) * (1-sigmoid(out))
    return out

def relu(x):
    out = np.copy(x)
    out[out < 0] = 0
    out[out > 1] = 1
    return out

def relu_deriv(x):
    out = np.copy(x)
    out[out < 0] = 0
    out[out > 1] = 0
    out[(out >= 0) & (out <= 1)] = 1
    return out

def leaky_relu(x):
    out = np.copy(x)
    out[out < 0] *= 0.01
    out[out > 1] = 1 + 0.01*(out[out > 1]-1)
    return out

def leaky_relu_deriv(x):
    out = np.copy(x)
    out[out < 0] = 0.01
    out[out > 1] = 0.01
    out[(out >= 0) & (out <= 1)] = 1
    return out

# Select activation function (sigmoid or relu)
activ_func = leaky_relu
activ_func_deriv = leaky_relu_deriv



class Network:
    # Initialization
    def __init__(self, c_in, c_out, random=0):

        # Numpy settings
        np.set_printoptions(precision=2, suppress=True) # numpy print settings


        # Training coefficient (CONST)
        self.ALPHA = 0.001


        # Mistakes (plotting)
        # Different types of mistakes analysis
        # corteg -- the average mistake value fpr the training array
        # plot 1: linear loss
        self.mists_lin = []
        self.mists_lin_corteg = []
        # plot 2: Square loss
        self.mists_sq = []
        self.mists_sq_corteg = []
        # plot 3: Cross Entropy loss
        self.mists_entr = []
        self.mists_entr_corteg = []


        # Neurons count (CONST)
        self.C_IN = c_in
        self.C_H1 = 500
        self.C_H2 = 250
        self.C_H3 = 100
        self.C_OUT = c_out


        # Layers of Neurons (vector)
        self.l_in = np.zeros((1, self.C_IN))
        self.l_out_right = np.zeros((1, self.C_OUT)) # For training

        self.l_h1 = np.zeros((1, self.C_H1))
        self.l_h2 = np.zeros((1, self.C_H2))
        self.l_h3 = np.zeros((1, self.C_H3))
        self.l_out = np.zeros((1, self.C_OUT))
        self.choice = -1


        # Weights (matrix)
        self.W_in_h1 = None
        self.W_h1_h2 = None
        self.W_h2_h3 = None
        self.W_h3_out = None


        # Shift (vector)
        self.s_in_h1 = None
        self.s_h1_h2 = None
        self.s_h2_h3 = None
        self.s_h3_out = None


        # File to collect arrays (with marks about layers size)
        self.arrays_filename = f"Arrays_{self.C_IN}_{self.C_H1}_{self.C_H2}_{self.C_H3}_{self.C_OUT}"
        self.arrays_dir = Path.cwd() / 'Arrays_Data'
        self.arrays_full_path = self.arrays_dir / self.arrays_filename
        # create folders
        if not self.arrays_dir.is_dir():
            self.arrays_dir.mkdir()


        # Check file exists
        if (self.arrays_full_path.with_suffix(".npz")).is_file() or random == 1:
            # load weights and shifts data from the file
            self.load_data()
        else:
            # Randomization weights and shifts
            self.W_in_h1 = np.random.uniform(-0.1, 0.1, (self.C_IN, self.C_H1))
            self.W_h1_h2 = np.random.uniform(-0.1, 0.1, (self.C_H1, self.C_H2))
            self.W_h2_h3 = np.random.uniform(-0.1, 0.1, (self.C_H2, self.C_H3))
            self.W_h3_out = np.random.uniform(-0.1, 0.1, (self.C_H3, self.C_OUT))

            self.s_in_h1 = np.random.uniform(-0.1, 0.1, (1, self.C_H1))
            self.s_h1_h2 = np.random.uniform(-0.1, 0.1, (1, self.C_H2))
            self.s_h2_h3 = np.random.uniform(-0.1, 0.1, (1, self.C_H3))
            self.s_h3_out = np.random.uniform(-0.1, 0.1, (1, self.C_OUT))


        # Mistakes
        self.m_h1 = np.zeros((1, self.C_H1))
        self.m_h2 = np.zeros((1, self.C_H2))
        self.m_h3 = np.zeros((1, self.C_H3))
        self.m_out = np.zeros((1, self.C_OUT))



    # Public funcs
    def run(self, _in):
        # Use AI for choosing next step
        # send input data

        # Setting input layer
        self.l_in = np.array(_in, dtype=float).reshape(1,_in.size)

        # Calculate output
        self.forward()

        # Choosing commad based on probability
        # self.choice = np.random.choice([-1,0,1], p=self.l_out[0])

        # Choosing command based on max value
        self.choice = np.argmax(self.l_out)

    def training(self, _in, _out):
        # AI training
        # using input and correct output data
        # (the correct output data is knows)

        # setting input and correct output layers
        self.l_in = np.array(_in, dtype=float).reshape(1,_in.size)
        self.l_out_right = np.array(_out).reshape(1,_out.size)

        # Calculate output, finding mistakes and correction matrixes
        self.forward()
        self.choice = np.argmax(self.l_out)
        self.backward()
        self.update()


    # Private funcs
    def forward(self):
        # Finding output
        self.l_h1_row = self.l_in @ self.W_in_h1 + self.s_in_h1
        self.l_h1 = activ_func(self.l_h1_row)
        self.l_h2_row = self.l_h1_row @ self.W_h1_h2 + self.s_h1_h2
        self.l_h2 = activ_func(self.l_h2_row)
        self.l_h3_row = self.l_h2_row @ self.W_h2_h3 + self.s_h2_h3
        self.l_h3 = activ_func(self.l_h3_row)
        self.l_out_row = self.l_h3_row @ self.W_h3_out + self.s_h3_out
        self.l_out = softmax(self.l_out_row)

    def backward(self):
        # Finding mistakes
        self.m_out = self.l_out - self.l_out_right
        self.m_h3 = self.m_out @ self.W_h3_out.T * activ_func_deriv(self.l_h3)
        self.m_h2 = self.m_h3 @ self.W_h2_h3.T * activ_func_deriv(self.l_h2)
        self.m_h1 = self.m_h2 @ self.W_h1_h2.T * activ_func_deriv(self.l_h1)

    def update(self):
        # Updating weights and shiftings
        self.W_in_h1 -= self.ALPHA * self.l_in.T @ self.m_h1
        self.s_in_h1 -= self.ALPHA * np.sum(self.m_h1, axis=0, keepdims=True)
        self.W_h1_h2 -= self.ALPHA * self.l_h1_row.T @ self.m_h2
        self.s_h1_h2 -= self.ALPHA * np.sum(self.m_h2, axis=0, keepdims=True)
        self.W_h2_h3 -= self.ALPHA * self.l_h2_row.T @ self.m_h3
        self.s_h2_h3 -= self.ALPHA * np.sum(self.m_h3, axis=0, keepdims=True)
        self.W_h3_out -= self.ALPHA * self.l_h3_row.T @ self.m_out
        self.s_h3_out -= self.ALPHA * np.sum(self.m_out, axis=0, keepdims=True)


    # Getters
    def get_choice(self):
        # Get choice from the other source
        return self.choice

    def get_output(self):
        # Get output layer values
        # after softmax function
        return self.l_out


    # Mistakes calculation with different modification
    def calc_mist_lin(self):
        mist = linear_loss(self.l_out, self.l_out_right)
        self.mists_lin.append(mist)

    def calc_mist_sq(self):
        mist = quadratic_loss(self.l_out, self.l_out_right)
        self.mists_sq.append(mist)

    def calc_mists_entr(self):
        # Works normally only with classification tasks
        # when in the correct output there is only one "1"
        # without less numbers, like 0.5, 0.33 etc.
        # [0, ..., 0, 1, 0, ..., 0]
        E = cross_entropy_loss(self.l_out, self.l_out_right)
        self.mists_entr.append(E)

    def calc_mist_corteg(self, last_count):
        # corteg -- the average mistake value for the training array
        mist_lin = 0
        mist_sq = 0
        mist_entr = 0
        for i in range(last_count):
            mist_lin += self.mists_lin[-i-1]
            mist_sq += self.mists_sq[-i-1]
            mist_entr += self.mists_entr[-i-1]
        mist_lin /= last_count
        mist_sq /= last_count
        mist_entr /= last_count
        self.mists_lin_corteg.append(mist_lin)
        self.mists_sq_corteg.append(mist_sq)
        self.mists_entr_corteg.append(mist_entr)


    # Debugging data
    def show_graphics(self):
            # Graphics settings
            figure, axes = plt.subplots(2, 2, figsize=(8.96, 6.72))
            axes[0,0].set_title('Linear')
            axes[0,1].set_title('Square')
            axes[1,0].set_title('Cross-entropy')
            axes[0,0].set_ylim([0, 1])
            axes[0,1].set_ylim([0, 1])
            axes[1,0].set_ylim([0, 1])
            axes[1,1].set_ylim([0, 1])

            # Check for the empty arrays
            if len(self.mists_sq_corteg) == 0:
                return

            # First graphic (linear analysis)
            axes[0,0].plot(self.mists_lin_corteg, 'r.', markeredgewidth = 0)

            axes[0,0].text(0.02, 0.94, f"start={'%.3f' % self.mists_lin_corteg[0]}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,0].text(0.5, 0.94, f"end={'%.3f' % self.mists_lin_corteg[-1]}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,0].text(0.02, 0.78, f"min={'%.3f' % min(self.mists_lin_corteg)}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,0].text(0.5, 0.78, f"max={'%.3f' % max(self.mists_lin_corteg)}",
                color = 'white', transform=axes[0,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Second graphic (square analysis)
            axes[0,1].plot(self.mists_sq_corteg, 'b.', markeredgewidth = 0)

            axes[0,1].text(0.02, 0.94, f"start={'%.3f' % self.mists_sq_corteg[0]}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,1].text(0.5, 0.94, f"end={'%.3f' % self.mists_sq_corteg[-1]}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,1].text(0.02, 0.78, f"min={'%.3f' % min(self.mists_sq_corteg)}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[0,1].text(0.5, 0.78, f"max={'%.3f' % max(self.mists_sq_corteg)}",
                color = 'white', transform=axes[0,1].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Third graphics (entropy analysis)
            axes[1,0].plot(self.mists_entr_corteg, 'y.', markeredgewidth = 0)

            axes[1,0].text(0.02, 0.94, f"start={'%.3f' % self.mists_entr_corteg[0]}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[1,0].text(0.5, 0.94, f"end={'%.3f' % self.mists_entr_corteg[-1]}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[1,0].text(0.02, 0.78, f"min={'%.3f' % min(self.mists_entr_corteg)}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})
            axes[1,0].text(0.5, 0.78, f"max={'%.3f' % max(self.mists_entr_corteg)}",
                color = 'white', transform=axes[1,0].transAxes,
                bbox={'facecolor': 'blue', 'alpha': 0.85, 'pad': 4})

            # Showing
            plt.show()

    def show_data(self):
        # Print weights and shifts into a consol
        print("\nW_in_h1\n", self.W_in_h1)
        print("\nW_h1_h2\n", self.W_h1_h2)
        print("\nW_h2_h3\n", self.W_h2_h3)
        print("\nW_h3_out\n", self.W_h3_out)
        print("\ns_in_h1\n", self.s_in_h1)
        print("\ns_h1_h2\n", self.s_h1_h2)
        print("\ns_h2_h3\n", self.s_h2_h3)
        print("\ns_h3_out\n", self.s_h3_out)


    # Data manipulation
    def save_data(self):
        print("!!!!!")
        # Save matrixes into the file
        np.savez(self.arrays_full_path,
                    self.W_in_h1, self.W_h1_h2, self.W_h2_h3, self.W_h3_out,
                    self.s_in_h1, self.s_h1_h2, self.s_h2_h3, self.s_h3_out)

    def load_data(self):
        # load matrixes from the file
        file = np.load(self.arrays_full_path.with_suffix(".npz"))
        self.W_in_h1 = file['arr_0']
        self.W_h1_h2 = file['arr_1']
        self.W_h2_h3 = file['arr_2']
        self.W_h3_out = file['arr_3']
        self.s_in_h1 = file['arr_4']
        self.s_h1_h2 = file['arr_5']
        self.s_h2_h3 = file['arr_6']
        self.s_h3_out = file['arr_7']



if __name__ == "__main__":

        c_in = 28*28
        c_out = 10

        network = Network(c_in, c_out)
        network.show_data()
