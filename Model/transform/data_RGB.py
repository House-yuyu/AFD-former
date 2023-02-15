import os
from transform.dataset_zx import Dataset_train, Dataset_val
# from transform.dataset_fusion import Dataset_train, Dataset_val


def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return Dataset_train(rgb_dir)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return Dataset_val(rgb_dir)


# def get_test_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTest_(rgb_dir, img_options)
