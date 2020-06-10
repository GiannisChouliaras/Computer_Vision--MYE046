import numpy as np
import cv2
import sys

circles = np.zeros((4, 2), np.int)
counter = 0


def coordinates(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        circles[counter] = x, y
        counter = counter + 1
    #
#


def myPerspectiveTransform(points1, points2):
    # make the 3 arrays : A , x, b
    # calculate A-¹ (inverse) and find solutions for x
    # with x = A-¹ * b
    x = np.zeros((8, 1))
    b = np.array([
        [0],
        [0],
        [points2[1][0]],
        [0],
        [0],
        [points2[2][1]],
        [points2[3][0]],
        [points2[3][1]]])
    A = np.array([
        [points1[0][0], points1[0][1], 1, 0, 0, 0, - points1[0]
            [0]*points2[0][0], - points1[0][1]*points2[0][0]],
        [0, 0, 0, points1[0][0], points1[0][1], 1, - points1[0]
            [0]*points2[0][1], - points1[0][1]*points2[0][1]],
        [points1[1][0], points1[1][1], 1, 0, 0, 0, -points1[1]
            [0]*points2[1][0], - points1[1][1]*points2[1][0]],
        [0, 0, 0, points1[1][0], points1[1][1], 1, - points1[1]
            [0]*points2[1][1], - points1[1][1]*points2[1][1]],
        [points1[2][0], points1[2][1], 1, 0, 0, 0, -points1[2]
            [0]*points2[2][0], - points1[2][1]*points2[2][0]],
        [0, 0, 0, points1[2][0], points1[2][1], 1, - points1[2]
            [0]*points2[2][1], - points1[2][1]*points2[2][1]],
        [points1[3][0], points1[3][1], 1, 0, 0, 0, -points1[3]
            [0]*points2[3][0], - points1[3][1]*points2[3][0]],
        [0, 0, 0, points1[3][0], points1[3][1], 1, - points1[3]
            [0]*points2[3][1], - points1[3][1]*points2[3][1]]])
    _A = np.linalg.inv(A)
    x = _A @ b
    result = np.array([
        [x[0][0], x[1][0], x[2][0]],
        [x[3][0], x[4][0], x[5][0]],
        [x[6][0], x[7][0], 1]])
    return result
#


def warp(img):
    global counter
    copy = img.copy()
    name = "Original Image"
    while True:
        if counter == 4:
            width, height = 1000, 1000  # 1000 x 1000 pixels
            points1 = np.float32(
                [circles[0], circles[1], circles[2], circles[3]])
            points2 = np.float32(
                [[0, 0], [width, 0], [0, height], [width, height]])
            myMatrix = myPerspectiveTransform(points1, points2)
            imgOutput = cv2.warpPerspective(img, myMatrix, (width, height))
            return imgOutput
        #

        # Make the circles in copy image.
        for x in range(0, 4):
            cv2.circle(copy, (circles[x][0], circles[x][1]), 6,
                       (45, 100, 45), cv2.FILLED)
        #
        cv2.namedWindow(name)
        cv2.moveWindow(name, 700, 250)
        cv2.imshow(name, copy)
        cv2.setMouseCallback(name, coordinates)
        cv2.waitKey(1)
#


if __name__ == "__main__":
    arguments = len(sys.argv)
    if not bool(arguments == 3):
        raise("Wrong values: Python3 warp.py <input-file> <outputfile>")
    #
    input_name = sys.argv[1]
    output_name = sys.argv[2]
    img = cv2.imread(input_name)
    array = warp(img)  # get the final image
    cv2.imwrite(output_name, array)  # write it
#
