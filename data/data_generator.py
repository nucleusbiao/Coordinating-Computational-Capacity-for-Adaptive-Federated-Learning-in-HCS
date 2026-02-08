from struct import *
import numpy as np
from utils import get_one_hot_from_label_index


def Apple_leaf_data_generator(sample_list, batch_size):
    """
    Generate batches of apple leaf data dynamically.
    This generator yields batches of training data without loading everything into memory.
    """
    BYTES_EACH_IMAGE = 150528
    BYTES_EACH_LABEL = 1
    BYTES_EACH_SAMPLE = BYTES_EACH_LABEL + BYTES_EACH_IMAGE
    SAMPLES_EACH_FILE = 1750

    f = []
    file_path = "datasets/plant-leaf-disease-224x224/train/train_data.bin"
    f.append(open(file_path, 'rb'))

    idx = 0
    sample_list = list(sample_list)
    total_samples = len(sample_list)

    while True:
        np.random.shuffle(sample_list)
        data = []
        labels = []
        indices = []
        
        for i in range(batch_size):
            sample_index = sample_list[idx]
            indices.append(sample_index)
            f[0].seek(sample_index * BYTES_EACH_SAMPLE)
            label = unpack('>B', f[0].read(1))[0]
            y = get_one_hot_from_label_index(label)
            x = np.array(list(f[0].read(BYTES_EACH_IMAGE)))
            
            tmp_mean = np.mean(x)
            tmp_stddev = np.std(x)
            tmp_adjusted_stddev = max(tmp_stddev, 1.0 / np.sqrt(len(x)))
            x = (x - tmp_mean) / tmp_adjusted_stddev
            x = np.reshape(x, [224, 224, 3], order='F')
            x = np.reshape(x, [150528], order='C')
            
            data.append(x)
            labels.append(y)
            idx = (idx + 1) % total_samples

        yield np.array(data), np.array(labels), indices
