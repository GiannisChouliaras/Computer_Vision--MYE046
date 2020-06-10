'''
    Homework_1 for the Course Computational Vision!  "Threshold an image"
    @Author    : Chouliaras Ioannis , AM : 2631
    @Professor : Sfikas Giorgos
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2020 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# Imports:
import sys
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

# Global variable for image.
global original_image
global edited_image


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ first step: Open the image ~~~~~~~~~
def open_image(filename):
    global original_image
    original_image = np.array(Image.open(filename))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF FIRST STEP FUNCTION ~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ middle step: Transform the image ~~~
def transform_with_threshold(value):
    global original_image
    global edited_image

    edit_image = np.copy(original_image)
    rows = edit_image.shape[0]
    cols = edit_image.shape[1]

    if len(edit_image.shape) == 2:
        for i in range(rows):
            for j in range(cols):
                if edit_image[i][j] > value:
                    edit_image[i][j] = 255
                else:
                    edit_image[i][j] = 0
    else:
        for i in range(rows):
            for j in range(cols):
                pos = edit_image[i][j][0] + edit_image[i][j][1] + \
                      edit_image[i][j][2]
                pos = pos / 3
                if pos > value:
                    edit_image[i][j][0] = 255
                    edit_image[i][j][1] = 255
                    edit_image[i][j][2] = 255
                else:
                    edit_image[i][j][0] = 0
                    edit_image[i][j][1] = 0
                    edit_image[i][j][2] = 0

    edited_image = Image.fromarray(edit_image)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF FIRST STEP FUNCTION ~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ last step: Save the image ~~~~~~~~~~~
def save_image(export_name):
    global edited_image
    edited_image.save(export_name)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF LAST STEP FUNCTION ~~~~~~~


def exit_function():
    print("Wrong values, must be 4\nFirst the name of the python file\n"
          "Second the name of the picture you want"
          " to edit\nThird the name of the exported file\nLast the K value")
    exit(-1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main Function ~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    arguments = len(sys.argv)
    correct = bool(arguments == 4)
    if (correct == False):  # Check the correct arguments
        exit_function()
    else:
        image_to_open_name = sys.argv[1]  # String with the image name to open
        image_to_save_name = sys.argv[2]  # String with the image name to save
        threshold_of_image = int(sys.argv[3])  # integer with the thres. value
        open_image(image_to_open_name)
        transform_with_threshold(threshold_of_image)
        save_image(image_to_save_name)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF MAIN ~~~~~~~~~~~~~~~~~~~
