from utils import get_index_from_one_hot_label
from apple_leaf_extractor import Apple_leaf_extract


def get_data():
    """
    Load training and testing data for apple leaf disease classification.
    Returns:
        train_image, train_label, test_image, test_label, train_label_orig
    """
    train_image, train_label = Apple_leaf_extract(0, 1750, True)
    test_image, test_label = Apple_leaf_extract(0, 500, False)

    train_label_orig = []
    for i in range(0, len(train_label)):
        label = get_index_from_one_hot_label(train_label[i])
        train_label_orig.append(label[0])

    return train_image, train_label, test_image, test_label, train_label_orig


def get_data_train_samples(samples_list):
    """
    Get training samples for specific indices.
    """
    from apple_leaf_extractor import Apple_leaf_extract_samples

    train_image, train_label = Apple_leaf_extract_samples(samples_list, True)

    return train_image, train_label
