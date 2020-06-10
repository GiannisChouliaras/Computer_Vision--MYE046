'''
    Homework_2 for the Course Computational Vision! "Threshold--Otsu"
    @Author    : Chouliaras Ioannis , AM : 2631
    @Professor : Sfikas Giorgos
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2020 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import sys
import numpy as np
from PIL import Image

global image_array
global final


def check_b(rows, cols, row, col, d):
    x = row - d
    y = row + d
    z = col - d
    w = col + d
    if x < 0:
        x = 0
    if z < 0:
        z = 0
    if y > rows:
        y = rows
    if w > cols:
        w = cols
    return (x, y, z, w)
#


def threshold_image(row, col, threshold):
    global array_image
    if image_array[row][col] < threshold:
        image_array[row][col] = 0
    else:
        image_array[row][col] = 255
#


def calc_obj_Otsu(array, k):
    first_part = array[array < k]
    second_part = array[array >= k]
    middle1 = np.mean(first_part)
    middle2 = np.mean(second_part)
    middle = np.mean(array.flatten())
    pi1 = len(first_part) / (len(first_part) + len(second_part))
    pi2 = len(second_part) / (len(first_part) + len(second_part))
    calc = pi1 * (middle1 - middle)**2 + pi2 * (middle2 - middle)**2
    return calc
#


def otsu_Algorithm(d):
    global image_array
    global final
    copied = np.copy(image_array)
    best_value = 0
    best_k = 0
    rows = image_array.shape[0]
    cols = image_array.shape[1]
    for row in range(rows):
        for col in range(cols):
            (x, y, z, w) = check_b(rows, cols, row, col, d)
            arr = np.array(copied[x:y+1, z:w+1])
            for i in range(1, 256):
                obj_otsu = calc_obj_Otsu(arr, i)
                if obj_otsu > best_value:
                    best_k = i
                    best_value = obj_otsu
            threshold_image(row, col, best_k)
    final = Image.fromarray(image_array)
#


# Main
if __name__ == "__main__":
    arguments = len(sys.argv)
    correct = bool(arguments == 4)

    if not correct:
        raise("Wrong Values: Python3 adaptive.py image.png exit.png window_size")
    #
    image = sys.argv[1]
    filename = sys.argv[2]
    window_size = int(sys.argv[3])
    if window_size < 0:
        raise("seriously? Window size should be positive.")
    #
    image_array = np.array(Image.open(image))

    # check if grayscale or rgb
    if len(image_array.shape) != 2:
        rows = image_array.shape[0]
        cols = image_array.shape[1]
        grayscale_image = np.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                value = image_array[row][col][0] + \
                    image_array[row][col][1] + image_array[row][col][2]
                value = int(value/2)
                grayscale_image[row][col] = value
            #
        #
        grayscale_image = grayscale_image.astype(np.uint8)
        image_array = np.copy(grayscale_image)
    #
    otsu_Algorithm(window_size)
    final.save(filename)
#
