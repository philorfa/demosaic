import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#install("numpy")
#install("scipy")
#install("tk")
#install("matplotlib")
#install("tqdm")
#install("pandas")
#install("scikit-learn")
#install("ann_visualizer")
#install("graphviz")
#install("tensorflow-addons")
#install("opencv-python")
#install("pyopencl")
#install("scikit-image")



## Envorinment with python==3.9
## conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
## pip install --upgrade pip /  C:\Users\philo\anaconda3\envs\tf\python.exe -m pip install --upgrade pip
## pip install "tensorflow<2.11"
## check import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))
## check import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

##In order to save and load models!!
##pip install pyyaml h5py