import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import tensorflow as tf
import keras


def count_nonzero(array):
    return np.count_nonzero(array)


def heatmap2d_antennas(arr: np.ndarray, antennas):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    length = len(antennas)
    for i, point in enumerate(antennas):
        plt.annotate((length - i) % length + 1, (point[1], point[0]), color='white')
        plt.scatter(point[1], point[0], marker="x", color="red", s=200)
    plt.gca().invert_yaxis()
    plt.show()


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()


def print_element(arr, element):
    print("!!!!!____________", element, " ELEMENT____________!!!!!\n")
    for x in arr[element]:
        print(x, "\ndtype:", x.dtype, "\n\n")


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    print("Target Frequency: ", value)
    print("Closest Frequency we found: ", array[idx])
    return idx


def read_file(text="Select file to open"):
    root = Tk()
    path = askopenfilename(title=text)
    root.update()
    return path


def save_directory(text="Select directory to save file"):
    root = Tk()
    path = askdirectory(title=text)
    root.update()
    return path


def column(matrix, i):
    return [row[i] for row in matrix]


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def keras_to_pb(model, output_filename, output_node_names):
    """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
      the function gets the last layer name as the output node.
   """

    # Get the names of the input and output nodes.
    in_name = model.layers[0].get_output_at(0).name.split(':')[0]

    if output_node_names is None:
        output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

    sess = keras.backend.get_session()

    # The TensorFlow freeze_graph expects a comma-separated string of output node names.
    output_node_names_tf = ','.join(output_node_names)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    sess.close()
    wkdir = ''
    tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

    return in_name, output_node_names
