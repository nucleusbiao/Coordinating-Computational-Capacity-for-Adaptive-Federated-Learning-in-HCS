from struct import *
import numpy as np
from utils import get_one_hot_from_label_index

BYTES_EACH_LABEL = 1
BYTES_EACH_IMAGE = 150528
BYTES_EACH_SAMPLE = BYTES_EACH_LABEL + BYTES_EACH_IMAGE


def Apple_leaf_extract_samples(sample_list, is_train=True):
    f_list = []
    if is_train:
        file_path = "datasets/plant-leaf-disease-224x224/train/train_data.bin"
        SAMPLES_EACH_FILE = 1750
    else:
        file_path = "datasets/plant-leaf-disease-224x224/test/test_data.bin"
        SAMPLES_EACH_FILE = 500
    
    f_list.append(open(file_path, 'rb'))
    data = []
    labels = []

    for i in sample_list:
        file_index = int(i / float(SAMPLES_EACH_FILE))
        f_list[file_index].seek((i - file_index * SAMPLES_EACH_FILE) * BYTES_EACH_SAMPLE)
        label = unpack('>B', f_list[file_index].read(1))[0]
        y = get_one_hot_from_label_index(label)
        x = np.array(list(f_list[file_index].read(BYTES_EACH_IMAGE)))

        tmp_mean = np.mean(x)
        tmp_stddev = np.std(x)
        tmp_adjusted_stddev = max(tmp_stddev, 1.0 / np.sqrt(len(x)))
        x = (x - tmp_mean) / tmp_adjusted_stddev

        x = np.reshape(x, [224, 224, 3], order='F')
        x = np.reshape(x, [150528], order='C')

        data.append(x)
        labels.append(y)

    for f in f_list:
        f.close()

    return data, labels


def Apple_leaf_extract(start_sample_index, num_samples, is_train=True):
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return Apple_leaf_extract_samples(sample_list, is_train)
