# Ioannis Chouliaras 2020 -- AM: 2631
# Second part of the final project of the course: Computer Vision
# Professor: Sfikas G.

import os
import sys
import tarfile
import tempfile
import cv2
import tensorflow as tf
import numpy as np
from six.moves import urllib
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes annoying warning.


class DeepLabModel(object):
    '''
    Here is the class of the model. I changes the OUTPUT_TENSOR_NAME to
    'concat_projection/Relu:0'
    Returns the resized image and seg_map.
    seg_map now is a tensor (_,_,256) where we will perform PCA
    '''
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'concat_projection/Relu:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map  # seg_map is the tensor, use that later for PCA


def get_model():
    '''
    With this function we download the pretrained model:
    'mobilenetv2_coco_voctrainaug'
    When we download it succesfully we return it to a variable
    for future use from main. We don't need to include the other URL's
    '''
    _MODEL_URL = 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _TARBALL_NAME = 'deeplab_model.tar.gz'
    model_dir = tempfile.mkdtemp()
    tf.gfile.MakeDirs(model_dir)
    download_path = os.path.join(model_dir, _TARBALL_NAME)
    print('downloading model, this might take a while...')
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URL,
                               download_path)
    print('download completed! loading DeepLab model...')
    model = DeepLabModel(download_path)
    print('model loaded successfully!')
    return model


def get_deep_feature(image, MODEL):
    ''' 
    In this function we will run the visualization. (MODEL.run)
    We will get the tensor with shape (_,_,256) where we can perform
    the PCA to drop the dimension to d = 3
    '''
    try:
        original_image = Image.open(image)
    except IOError:
        print('Cannot retrieve image. Please check the name')
        return
    print('running deeplab on image %s...' % image)
    return MODEL.run(original_image)  # returns reshaped_image, seg_map


def perform_pca(deepfeats):
    '''
    In this function, we perform PCA to drop the dimension of the tensor to d = 3
    Code was given from the professor on github [https://github.com/dip-course/pca_on_deepfeatures]
    A tensor with H x W x C, we reshape it to an array of HW x C (pixels x dimension of data)
    '''
    N = deepfeats.shape[0]*deepfeats.shape[1]
    C = deepfeats.shape[-1]
    X = np.reshape(deepfeats, [N, C])
    print('Τα αρχικά δεδομένα μου έχουν μέγεθος: {}'.format(X.shape))
    Xreduced = PCA(n_components=3).fit_transform(X)
    print('Μετά το PCA έχουμε μέγεθος: {}'.format(Xreduced.shape))
    return Xreduced


def visualization(after_pca, deepfeats, name):
    deepfeats_reduced = np.reshape(
        after_pca, [deepfeats.shape[0], deepfeats.shape[1], 3])
    print(deepfeats_reduced.shape)
    cv2.imwrite(name, deepfeats_reduced)


if __name__ == "__main__":
    '''
    Main: Run from terminal : python cv_second.py <input-name: 'example.jpg/png>
    get the pretrained model. For this part we download the model:
    'mobilenetv2_coco_voctrainaug' and get as result the deep feature:
    'concat_projection/Relu:0'
    Then perform PCA to the tensor for dropping the dimension to d = 3
    Finally save the image using cv2.
    '''
    arguments = len(sys.argv)
    if not bool(arguments == 3):
        raise("Wrong values, try: python cv_second.py <input image.jpg/png> <output image.jpg/png>")
    image = sys.argv[1]
    output_name = sys.argv[2]
    MODEL = get_model()
    # we don't need the reshaped image
    _, deep_feature = get_deep_feature(image, MODEL)
    after_pca = perform_pca(deep_feature)
    visualization(after_pca, deep_feature, output_name)
