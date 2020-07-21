# Ioannis Chouliaras 2020 -- AM: 2631
# Third part of the final project of the course: Computer Vision
# Professor: Sfikas G.

# For this part i will use the deep features from the second part. I will upload the npy arrays.

import os
import sys
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def main():
    arguments = len(sys.argv)
    if not bool(arguments == 2):
        raise("Wrong values, try: python cv_third.py image.jpg")
    image_name = sys.argv[1]
    load_array = get_deepfeats(image_name)
    deep_feature = np.load(load_array)
    deepfeats_reduced = perform_pca(deep_feature)
    final_array = perform_kmeans(deepfeats_reduced, deep_feature)
    visualize(final_array)


# end of main

def perform_pca(deepfeats):
    '''
    In this function, we perform PCA to drop the dimension of the tensor to d = 8
    Code was given from the professor on github [https://github.com/dip-course/pca_on_deepfeatures]
    A tensor with H x W x C, we reshape it to an array of HW x C (pixels x dimension of data)
    '''
    N = deepfeats.shape[0]*deepfeats.shape[1]
    C = deepfeats.shape[-1]
    X = np.reshape(deepfeats, [N, C])
    print('Τα αρχικά δεδομένα μου έχουν μέγεθος: {}'.format(X.shape))
    Xreduced = PCA(n_components=8).fit_transform(X)
    print('Μετά το PCA έχουμε μέγεθος: {}'.format(Xreduced.shape))
    return Xreduced


def get_deepfeats(image):
    '''
        Just take the correct npy file from disk using the image input from terminal.
    '''
    deepfeats = {
        'city.jpg': 'city.npy',
        'dining.jpg': 'dining.npy',
        'horse_man.jpg': 'horse_man.npy',
        'office.jpg': 'office.npy',
        'plane_train.jpg': 'plane_train.npy',
    }
    return deepfeats[image]


def perform_kmeans(deepfeats, df_original_size):
    '''
    Here i will perform the K-means algorithm with number of clusters = 2. That will return
    a binarization of the image where we separate background and point of interest.
    '''
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(deepfeats)
    flatten_labels = kmeans.labels_
    print('The shape of centers is {}'.format(kmeans.cluster_centers_.shape))
    print('The shape of labels is {}'.format(flatten_labels.shape))
    # reshape the labels as the deep feature shape
    final_array = flatten_labels.reshape(
        (df_original_size.shape[0], df_original_size.shape[1]))
    return final_array


def change_array(array):
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            if array[i][j] == 1:
                array[i][j] = 255
    return array.astype(np.uint8)


def visualize(array):
    '''
    Visualize the array using cv2 as in second exercise of the course
    '''
    final = change_array(array)
    cv2.namedWindow('segmentation PCA - Kmeans', cv2.WINDOW_NORMAL)
    cv2.imshow('segmentation PCA - Kmeans', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
