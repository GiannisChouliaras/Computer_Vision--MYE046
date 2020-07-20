# Ioannis Chouliaras 2020 -- AM: 2631
# First part of the final project of the course: Computer Vision
# Professor: Sfikas G.

import os
import sys
import tensorflow as tf
import tarfile
import tempfile
from six.moves import urllib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # annoying warning.


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
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
        return resized_image, seg_map


def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    unique_labels = np.unique(seg_map)
    colors = FULL_COLOR_MAP[unique_labels].astype(np.uint8)
    labels_names = LABEL_NAMES[unique_labels]
    return (seg_image, colors, labels_names)


def get_model():
    MODEL_NAME = 'mobilenetv2_coco_voctrainval'
    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URL = 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
    _TARBALL_NAME = 'deeplab_model.tar.gz'

    model_dir = tempfile.mkdtemp()
    tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    print('downloading model, this might take a while...')
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URL,
                               download_path)
    print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')
    return MODEL


def create_labels():
    label_names = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])
    full_label_map = np.arange(len(label_names)).reshape(len(label_names), 1)
    full_color_map = label_to_color_image(full_label_map)
    return (label_names, full_label_map, full_color_map)


def run_visualization(image_name, output_name, MODEL):
    try:
        original_im = Image.open(image_name)
    except IOError:
        print('Cannot retrieve image. Please check the name')
        return
    print('running deeplab on image %s...' % image_name)
    resized_im, seg_map = MODEL.run(original_im)
    (segm_image, colors, labels) = vis_segmentation(resized_im, seg_map)
    final_image = create_final_image(segm_image, colors, labels)
    final_image.save(output_name)


def create_final_image(array_image, colors, labels):
    w_prev = 0
    image = Image.fromarray(array_image)
    width, height = image.size
    n_image = Image.new('RGB', (width+10, int(height+(height/5))), 'white')
    n_image.paste(image, (5, 5, (width + 5), (height+5)))
    return draw_image(n_image, colors, labels, width, height)


def draw_image(image, colors, labels, width, height):
    font = ImageFont.truetype("/Users/mac/Library/Fonts/BEBAS.ttf", 20)
    draw = ImageDraw.Draw(image)
    previous_width = 0
    for i in range(len(labels)):
        label = labels[i]
        current_w, current_h = font.getsize(label)
        if i == 0:
            draw_w = (width-current_w)
        else:
            draw_w = (width-current_w) - previous_width
        draw_h = (height + ((height/5)-current_h)/2)
        color = (colors[i][0][0], colors[i][0][1], colors[i][0][2])
        draw.text((draw_w, draw_h), label, font=font, fill=color)
        previous_width += current_w
    #
    return image


if __name__ == "__main__":
    arguments = len(sys.argv)
    if not bool(arguments == 3):
        raise("Wrong values: python3 warp.py <input-file> <output-file>")
    input_name = sys.argv[1]
    output_name = sys.argv[2]
    (LABEL_NAMES, FULL_LABEL_MAP, FULL_COLOR_MAP) = create_labels()
    MODEL = get_model()
    run_visualization(input_name, output_name, MODEL)
# end of main
