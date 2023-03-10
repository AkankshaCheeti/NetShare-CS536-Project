import enum
import os
import math
import copy
from more_itertools import sample
import numpy as np
from tqdm import tqdm

import netshare.ray as ray

# from ray.util.multiprocessing import Pool
# from ray.util.queue import Queue

from .util import add_gen_flag, normalize_per_sample
from multiprocess import Pool, Value, Manager, Queue


class NetShareDataset(object):
    def __init__(
        self,
        root,
        config,
        data_attribute_outputs,
        data_feature_outputs,
        buffer_size=1000,
        num_processes=10,
        *args,
        **kwargs
    ):
        self.config = config
        self.data_attribute_outputs_orig = data_attribute_outputs  # immutable
        self.data_feature_outputs_orig = data_feature_outputs  # immutable
        self.data_attribute_outputs_train = None  # mutable, feed in training
        self.data_feature_outputs_train = None  # mutable, feed in training
        self.real_attribute_mask = None  # mutable, feed in training
        self.gt_lengths = None  # mutable, feed in training
        super(NetShareDataset, self).__init__(*args, **kwargs)

        self.root = root
        self.buffer_size = buffer_size
        self.num_processes = num_processes

        self.manager = Manager()
        self.running_flag = self.manager.Value("i", 1)
        self.image_buffer = self.manager.Queue(maxsize=buffer_size)
        self.files = self.manager.list(
            [
                os.path.join(root, "data_train_npz", file)
                for file in os.listdir(os.path.join(root, "data_train_npz"))
                if file.endswith(".npz")
            ]
        )
        self.config_mp = self.manager.dict(config)

        self.pool = Pool(num_processes)
        print("prepared to start data loader")
        self.results = [
            self.pool.apply_async(
                self.data_loader,
                (
                    self.image_buffer,
                    self.running_flag,
                    self.files,
                    self.transform,
                    self.config_mp,
                ),
            )
            for _ in range(num_processes)
        ]

    @staticmethod
    def data_loader(image_buffer, running_flag, files, transform, config):
        np.random.seed(os.getpid())
        print("In data loader")
        print("-------------")
        while running_flag.value == 1:
            file_id = np.random.choice(len(files))
            image = np.load(files[file_id])
            image_ = {}
            for k in image.files:
                image_[k] = image[k]
            image_ = transform(image_, config)
            try:
                image_buffer.put(image_, block=False)
            except BaseException:
                pass

        print("data loader ended")
        print("-------------")
        return True

    # append to global_max_flow_len
    @staticmethod
    def transform(image, config):
        image_ = {}

        data_attribute = image["data_attribute"]
        data_feature = image["data_feature"]
        data_gen_flag = image["data_gen_flag"]

        # pad to multiple of sample_len
        max_flow_len = image["global_max_flow_len"][0]
        ceil_timeseries_len = (
            math.ceil(
                max_flow_len / config["sample_len"]) * config["sample_len"]
        )
        data_feature = np.pad(
            data_feature,
            pad_width=(
                (0, ceil_timeseries_len - data_feature.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        data_gen_flag = np.pad(
            data_gen_flag,
            pad_width=(0, ceil_timeseries_len - data_gen_flag.shape[0]),
            mode="constant",
            constant_values=0,
        )

        image_["data_attribute"] = data_attribute
        image_["data_feature"] = data_feature
        image_["data_gen_flag"] = data_gen_flag

        return image_

    def stop_data_loader(self):
        print("Stop")
        self.running_flag.value = 0
        print(self.running_flag.value)
        for idx, res in enumerate(self.results):
            try:
                print(f"Stop data_loader #{idx}: {res.get(timeout=5)}")
            except:
                print(f"Stop data_loader #{idx} failed within timeout!")

        print("-------------")

        self.pool.close()
        self.pool.join()

    def sample_batch(self, batch_size):
        data_attribute = []
        data_feature = []
        data_gen_flag = []
        # Don't use busy waiting to check if queue is full.
        print(self.image_buffer.qsize())
        for i in range(batch_size):
            image = self.image_buffer.get()
            data_attribute.append(image["data_attribute"])
            data_feature.append(image["data_feature"])
            data_gen_flag.append(image["data_gen_flag"])

        data_attribute = np.stack(data_attribute, axis=0)
        data_feature = np.stack(data_feature, axis=0)
        data_gen_flag = np.stack(data_gen_flag, axis=0)

        if self.config["self_norm"]:
            (
                data_feature,
                data_attribute,
                self.data_attribute_outputs_train,
                self.real_attribute_mask,
            ) = normalize_per_sample(
                data_feature,
                data_attribute,
                data_gen_flag,
                self.data_feature_outputs_orig,
                self.data_attribute_outputs_orig,
            )
        else:
            self.real_attribute_mask = [True] * \
                len(self.data_attribute_outputs_orig)
            self.data_attribute_outputs_train = copy.deepcopy(
                self.data_attribute_outputs_orig
            )

        sample_len = self.config["sample_len"]
        if self.config["use_gt_lengths"]:
            self.data_feature_outputs_train = copy.deepcopy(
                self.data_feature_outputs_orig
            )
            self.gt_lengths = np.load(
                os.path.join(
                    self.root,
                    "gt_lengths.npy"))
        else:
            data_feature, self.data_feature_outputs_train = add_gen_flag(
                data_feature, data_gen_flag, self.data_feature_outputs_orig, sample_len
            )
            self.gt_lengths = None

        data_gen_flag = np.expand_dims(data_gen_flag, 2)

        return data_attribute, data_feature, data_gen_flag

    def sample_batch_with_rowid(self, batch_size):
        data_attribute = []
        data_feature = []
        data_gen_flag = []
        data_row_ids = []
        for i in range(batch_size):
            image = self.image_buffer.get()
            data_attribute.append(image["data_attribute"])
            data_feature.append(image["data_feature"])
            data_gen_flag.append(image["data_gen_flag"])
            data_row_ids.append(image["row_id"])

        data_attribute = np.stack(data_attribute, axis=0)
        data_feature = np.stack(data_feature, axis=0)
        data_gen_flag = np.stack(data_gen_flag, axis=0)

        # print(data_attribute.shape)
        # print(data_feature.shape)
        # print(data_gen_flag.shape)

        if self.config["self_norm"]:
            (
                data_feature,
                data_attribute,
                self.data_attribute_outputs_train,
                self.real_attribute_mask,
            ) = normalize_per_sample(
                data_feature,
                data_attribute,
                data_gen_flag,
                self.data_feature_outputs_orig,
                self.data_attribute_outputs_orig,
            )
        else:
            self.real_attribute_mask = [True] * \
                len(self.data_attribute_outputs_orig)
            self.data_attribute_outputs_train = copy.deepcopy(
                self.data_attribute_outputs_orig
            )

        sample_len = self.config["sample_len"]
        if self.config["use_gt_lengths"]:
            self.data_feature_outputs_train = copy.deepcopy(
                self.data_feature_outputs_orig
            )
            self.gt_lengths = np.load(
                os.path.join(
                    self.root,
                    "gt_lengths.npy"))
        else:
            data_feature, self.data_feature_outputs_train = add_gen_flag(
                data_feature, data_gen_flag, self.data_feature_outputs_orig, sample_len
            )
            self.gt_lengths = None

        data_gen_flag = np.expand_dims(data_gen_flag, 2)

        return data_attribute, data_feature, data_gen_flag, data_row_ids


# DO NOT USE THIS A TEST AS ``from .util import XX'' WILL CAUSE ERROR
# if __name__ == "__main__":
