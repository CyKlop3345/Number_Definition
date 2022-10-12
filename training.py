from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from time import time

from network import Network
from constants import *

def show_data():

    plt.figure(figsize=(10,5))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[i], cmap=plt.cm.binary)

    plt.show()



if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True, linewidth=250) # numpy print settings

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # show_data()

    # print(x_train[0])
    # print(y_train[0])

    c_in = 28*28
    c_out = 10

    network = Network(c_in, c_out)



    # Training
    # '''
    data_num = 60000
    loop_num = 100

    for _ in range(loop_num):
        time_start = time()
        for i in range(data_num):

            in_train = abs(x_train[i] / 255)

            out_train = np.zeros(10)
            out_train[y_train[i]] = 1

            network.training(in_train, out_train)
            network.calc_mist_lin()
            network.calc_mist_sq()
            network.calc_mists_entr()


            # in_train = abs(in_train-1)
            #
            # network.training(in_train, out_train)
            # network.calc_mist_lin()
            # network.calc_mist_sq()
            # network.calc_mists_entr()

        print(f'{100*(_+1)/loop_num :.1f} %\t  Time has passed: {(time() - time_start):.3f} sec.')     # Traning completion percentages

        network.calc_mist_corteg(data_num)



    # network.show_data()
    network.save_data()

    network.show_graphics()

    # network.show_data()

    # show_data()
    # '''



    # Testing
    # '''
    data_num = 500
    correct_num = 0
    for i in range(data_num):

        in_test = abs(x_test[i] / 255)
        # in_test = np.roll(in_test, randint(-3, 3), axis=randint(0, 1))

        network.run(in_test)
        subStr = ""
        if network.get_choice() != y_test[i]:
            subStr = "!!! incorrect !!!"
        else:
            correct_num += 1

        # print(f"{network.get_choice()}, {y_test[i]}\t{subStr}")


        # in_test = (in_test-1) * (-1)
        # network.run(in_test)
        # print(f"{network.get_choice()}, {y_test[i]}, Invert")

    print(f"Total: {correct_num}/{data_num}")

    # '''
