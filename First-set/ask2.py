'''
    Homework_1 for the Course Computational Vision!  "Affine Transforms"
    @Author    : Chouliaras Ioannis , AM : 2631
    @Professor : Sfikas Giorgos
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2020 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import numpy as np
from PIL import Image

global original_image
global edited_image


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ find neighbors function ~~~~~~~~~~~~

def find_neigbors(new_image_array):
    helping_array = np.copy(new_image_array)

    rows, cols = new_image_array.shape
    for row in range(rows):
        for col in range(cols):

            if helping_array[row][col] != -10:
                continue
            #
            else:
                i = 1
                while True:
                    x = row - i
                    y = col - i
                    z = row + i
                    w = col + i

                    # edges
                    if row - i < 0:
                        x = 0
                    if col - i < 0:
                        y = 0
                    if row + i > rows - 1:
                        z = rows - 1
                    if col + i > cols - 1:
                        w = cols - 1

                    # check neighbors
                    if helping_array[row][y] != -10:
                        new_image_array[row][col] = helping_array[row][y]
                        break
                    elif helping_array[row][w] != -10:
                        new_image_array[row][col] = helping_array[row][w]
                        break
                    elif helping_array[x][col] != -10:
                        new_image_array[row][col] = helping_array[x][col]
                        break
                    elif helping_array[z][col] != -10:
                        new_image_array[row][col] = helping_array[z][col]
                        break
                    elif helping_array[x][y] != -10:
                        new_image_array[row][col] = helping_array[x][y]
                        break
                    elif helping_array[x][w] != -10:
                        new_image_array[row][col] = helping_array[x][w]
                        break
                    elif helping_array[z][y] != -10:
                        new_image_array[row][col] = helping_array[z][y]
                        break
                    elif helping_array[z][w] != -10:
                        new_image_array[row][col] = helping_array[z][w]
                        break
                    else:
                        i = i + 1
                    # check neighbors
                # while
            # if - else
        # for col
    # for row

    return new_image_array


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF FIND NEIGBORS FUNCTION ~~~


def exit_function():
    print("Wrong values broda\nMust be 9")
    exit(-1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ first step: Open the image ~~~~~~~~~
def open_image_method(filename):
    global original_image
    original_image = np.array(Image.open(filename))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF FIRST STEP FUNCTION ~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ middle step: Affine the image ~~~~~~
def do_work_method(a, b, c, d, e, f):
    global original_image
    global edited_image

    # center
    rows, cols = original_image.shape
    middle_rows = int(rows / 2)
    middle_cols = int(cols / 2)

    full_size = rows * cols
    big_array = np.zeros((3, full_size))
    big_array[2, :] = 1

    i = 0
    for col in range(cols):
        for row in range(rows):
            big_array[0][i] = int(row - middle_rows)
            big_array[1][i] = int(col - middle_cols)
            i = i + 1
        #
    #

    affine_array = np.array([
        [float(a), float(b), float(c)],
        [float(d), float(e), float(f)],
        [0, 0, 1]
    ])

    new_big_array = affine_array @ big_array

    new_image_array = np.full((rows, cols), -10)
    i = 0

    for row in range(rows):
        for col in range(cols):
            x = new_big_array[0][i] + middle_rows
            y = new_big_array[1][i] + middle_cols

            if int(x) > rows - 1 or int(x) < 0 or int(y) > cols - 1 or int(y) < 0:
                i = i + 1
                continue
            #

            new_image_array[row][col] = original_image[int(y)][int(x)]
            i = i + 1
        #
    #

    # find neighbors
    new_image_array = find_neigbors(new_image_array)

    edited_image = Image.fromarray(new_image_array.astype(np.uint8))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF MIDDLE STEP FUNCTION ~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ last step: Save the image ~~~~~~~~~~
def save_image_method(filename):
    global edited_image
    edited_image.save(filename)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF LAST STEP FUNCTION ~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    arguments = len(sys.argv)
    correct = bool(arguments == 9)
    if not correct:
        exit_function()
    else:
        image_open = sys.argv[1]
        image_save = sys.argv[2]
        a1 = sys.argv[3]
        a2 = sys.argv[4]
        a3 = sys.argv[5]
        a4 = sys.argv[6]
        a5 = sys.argv[7]
        a6 = sys.argv[8]
        open_image_method(image_open)
        do_work_method(a1, a2, a3, a4, a5, a6)
        save_image_method(image_save)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF MAIN ~~~~~~~~~~~~~~~~~~~
