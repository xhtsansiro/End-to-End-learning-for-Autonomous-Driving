"""This script loads the data, for CARLA, LGSVL."""

# import the data for the training all models
import os
import bisect
import pandas as pd
from PIL import Image
import numpy as np
import torch
from utils import _augment_single_image_with_shadow

# DIR = "/Volumes/TOSHIBA EXT/Town01-03_train"

PIC_FOLDER = "rgb/data"

###############################################################################


# ZurichData, format of img changed to hdf5, for RNN,
# used to train NCP with EventData

class MarkovProcessRNNHdf5(torch.utils.data.Dataset):
    """This class is used for RNN, read hdf5 file."""

    def __init__(self, hf5, time_step, output=3,
                 transform=None, mode='train'):
        """Initialize the object."""
        # hf5 = hpy5.open("path'), <class 'h5py._hl.files.File'>
        self.hf5 = hf5
        # decides predicting steering or all actions
        self.output = output

        # in train mode, do the pic. augmentation,
        # in eval mode, do no do pic. augmentation.
        self.mode = mode
        self.time_step = time_step  # the sequence of the input
        self.sequence_size = []

        # contains the cumulative_size of the imgs
        self.cumulative_size = []

        cumulative_size = 0
        # get all the sequence name

        # how many sequences in this hdf5 group
        self.sequence_number = len(list(self.hf5.keys()))

        for i in range(self.sequence_number):
            # go to pic. group to see how many pics are in one sequence
            key1 = "sequence" + str(i) + "/pics"
            # go to pic. group to see how many pics are in one sequence
            key2 = "sequence" + str(i) + "/actions"
            length = len(list(self.hf5[key1].keys()))

            # pics should match acts
            assert length == len(list(
                self.hf5[key2].keys())), \
                "number of actions does not match number of pics"

            # save how many pics in each sequence.
            self.sequence_size.append(length)
            # how many pics are cumulated
            cumulative_size = \
                cumulative_size + len(list(self.hf5[key1].keys())) \
                - (self.time_step - 1)

            self.cumulative_size.append(cumulative_size)

        # how many pics in total
        self.n_data = \
            cumulative_size + (self.time_step - 1) * len(self.cumulative_size)

        self.total = cumulative_size  # how many batch samples are possible

        self.dtype = np.uint8  # more often used to store the pic.
        self.transform = transform

        print(f"how many sequences: {len(self.cumulative_size)}")
        print(f"number of data: {self.n_data}")
        print(f"observation type is {self.dtype}")

    def __len__(self):
        """Return the length of the dataset."""
        return self.total  # return the number of total training samples

    def __getitem__(self, idx):
        """Take state-action pairs from the dataset."""
        # the pics are in dictionary ["sequence1/pics/pic0"],
        # the actions are in dictionary ["sequence1/actions/act0"]
        s_o, a_c = [], []  # state observation, action command

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index "
                                 "should not exceed dataset length")
            idx = len(self) + idx

        if idx < len(self):  # len(self) is calling __len__(self)
            dataset_idx = bisect.bisect_right(self.cumulative_size, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_size[dataset_idx - 1]

            grp = "sequence" + str(dataset_idx)
            # a list contains all the keys of the images

            for j in range(self.time_step):
                img_key = grp + "/pics/pic" + str(sample_idx + j)  # [0,15]
                act_key = grp + "/actions/act" + str(sample_idx + j)

                # get the array from the hdf5 file
                img = np.array(self.hf5[img_key], dtype=self.dtype)
                # after getting picture as an array, 50% to add the shadow,
                # only in train mode
                if np.random.rand() > 0.5 and self.mode == 'train':
                    # adding the shadow, pic augmentation
                    # gamma from (1/1.5, 1.5), need an array of type uint8
                    img = _augment_single_image_with_shadow(img, 0.5)

                if self.output == 1:
                    act_array = np.array(self.hf5[act_key],
                                         dtype=np.float32)[0]
                    # use [0] when we just need steering
                else:
                    act_array = np.array(self.hf5[act_key],
                                         dtype=np.float32)

                if self.transform is not None:
                    # do the image transformation, after transformation,
                    # it will be tensor
                    img = self.transform(img)

                s_o.append(img)
                a_c.append(act_array)

            # after having all tensors in the so list,
            # change the list to the tensor
            s_o = torch.stack(s_o)  # (16,3,128,256)
            # convert list of array into an array,
            # if only steering, then one dimension np array
            a_c = np.array(a_c)
            a_c = torch.tensor(a_c)  # convert array to tensor (16,3)
            if self.output == 1:
                a_c = torch.unsqueeze(a_c, dim=1)
                # (16,) to (16,1), only for steering.
        return s_o, a_c

    # for sequence, for each sequence, iterate over the whole sequence.
    def iterate_as_single_sequence(self, sequence_number):
        """Load data for validation and test."""
        # sequence_number, which sequence to take to valid

        # from which sequence to take samples
        grp = "sequence" + str(sequence_number)

        # how many chunks does the taken sequence hat
        number_of_chunks = \
            self.sequence_size[sequence_number] // self.time_step

        for i in range(number_of_chunks):
            # take the index of one chunk,
            index = range(i*self.time_step, (i+1)*self.time_step)

            # a list contains all the keys of the images
            pic, actions = [], []
            for j in index:
                img_key = grp + "/pics/pic" + str(j)
                act_key = grp + "/actions/act" + str(j)
                img_array = np.array(self.hf5[img_key], dtype=self.dtype)
                if self.output == 1:
                    act_array = np.array(self.hf5[act_key],
                                         dtype=np.float32)[0]
                    # add [0] when it is only for steering,
                else:
                    act_array = np.array(self.hf5[act_key],
                                         dtype=np.float32)

                if self.transform is not None:
                    image = self.transform(img_array)

                # list of tensors
                pic.append(image)
                actions.append(act_array)

            picture = torch.stack(pic)  # (16,3,128,256)
            # convert list of array into an array
            actions = np.array(actions)
            actions = torch.tensor(actions)  # convert array to tensor (16,3)
            # for the case only steering:
            if self.output == 1:
                actions = torch.unsqueeze(actions, dim=1)
                # (16,) to (16,1)
            yield picture, actions  # return the tensor and the actions.


class MarkovProcessCNNHdf5(torch.utils.data.Dataset):
    """This class is used for CNN, read hdf5 file."""

    def __init__(self, hf5, output=3, transform=None, mode='train'):
        """Initialize the object."""

        self.hf5 = hf5
        # 1: only steering, 3: throttle, steering, brake
        self.output = output
        # in train mode, do the picture augmentation,
        # in eval mode, no pic augmentation.
        self.mode = mode

        self.sequence_size = []
        self.cumulative_size = []
        cumulative_size = 0
        # get all the sequence name

        # how many sequences in this hdf5 group
        self.sequence_number = len(list(self.hf5.keys()))

        for i in range(self.sequence_number):
            # go to pic. group to see how many pics are in one sequence
            key1 = "sequence" + str(i) + "/pics"
            # go to pic. group to see how many pics are in one sequence
            key2 = "sequence" + str(i) + "/actions"
            length = len(list(self.hf5[key1].keys()))
            # pics should match acts
            assert length == len(list(self.hf5[key2].keys())), \
                "number of actions does not match number of pics"

            # save how many pics in each sequence.
            self.sequence_size.append(length)
            cumulative_size += length
            # how many pics cumulatively
            self.cumulative_size.append(cumulative_size)

        # how many pictures in total
        self.n_data = sum(self.sequence_size)

        self.dtype = np.uint8  # more often used to store the pic.
        self.transform = transform

        print(f"how many sequences: {len(self.sequence_size)}")
        print(f"number of data: {self.n_data}")
        print(f"observation type is {self.dtype}")

    def __len__(self):
        """Return the length of Dataset."""
        return self.n_data  # return the number of total training samples

    def __getitem__(self, idx):
        """Take state-action pairs from the dataset."""
        # the pics are in dictionary ["sequence1/pics/pic0"],
        # the actions are in dictionary ["sequence1/actions/act0"]
        s_o, a_c = None, None

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should "
                                 "not exceed dataset length")
            idx = len(self) + idx

        if idx < len(self):  # len(self) is calling __len__(self)
            dataset_idx = bisect.bisect_right(self.cumulative_size, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_size[dataset_idx - 1]

            grp = "sequence" + str(dataset_idx)
            # a list contains all the keys of the images

            # for j in range(self.time_step):
            img_key = grp + "/pics/pic" + str(sample_idx)
            act_key = grp + "/actions/act" + str(sample_idx)

            # get the array from the hdf5 file
            img = np.array(self.hf5[img_key], dtype=self.dtype)
            # after getting picture as an array, 50% to add the shadow,
            # only in train mode
            if np.random.rand() > 0.5 and self.mode == 'train':
                # add the shadow, pic augmentation
                # gamma from (1/1.5, 1.5), need an array of type uint8
                img = _augment_single_image_with_shadow(img, 0.5)

            if self.output == 1:
                act_array = np.array(self.hf5[act_key],
                                     dtype=np.float32)[0]
                # only take the steering, which is the first value.
            else:
                act_array = np.array(self.hf5[act_key],
                                     dtype=np.float32)

            if self.transform is not None:
                s_o = self.transform(img)

            a_c = torch.tensor(act_array)  # convert array to tensor (1,3)
            if self.output == 1:
                a_c = torch.unsqueeze(a_c, dim=0)
            # for the case, only predict steering,

        return s_o, a_c


# ZurichData, no format change of img. for RNN series,
# used to train NCP with EventData
class MarkovProcessRNNPng(torch.utils.data.Dataset):
    """This class is used for RNN, read image in original format."""

    def __init__(self, dir_name, time_step, transform=None, mode='train'):
        """Initialize the object."""
        self.time_step = time_step    # the sequence of the input

        # list containing the list of each sequence for action
        self.action_all_sequence = []

        self.img_all_sequence = []   # ... for input: img_path,

        # contains the cumulative_size of the imgs
        self.cumulative_size = []

        self.mode = mode
        counter_output = 0
        cumulative_size = 0
        # read the observation
        # store all the paths to each image,steering,throttle, brake
        self.images, steering, throttle, brake = [], [], [], []

        data_town = os.listdir(dir_name)
        # get all the subfolder name, in MacOS, there is a invisiable DS.file
        data_town = list(filter(lambda a: '.DS_Store' not in a, data_town))
        # use filter to remove .DS_Store file in the list,
        print(data_town)
        print(f"No. of Towns: {len(data_town)}")
        for i in range(0, len(data_town)):
            name_town = data_town[i]
            subfolder = os.path.join(dir_name, name_town)
            # e.g /Volumes/TOSHIBA EXT/Town01-03_train/Town01/
            sequence = os.listdir(subfolder)
            sequence = list(filter(lambda a: '.DS_Store' not in a, sequence))

            for j in sequence:
                img_path, steering, throttle, brake = [], [], [], []
                # reset it at each new sequence
                rgb_folder = os.path.join(subfolder, j, PIC_FOLDER)

                for _, _, pic_names in sorted(os.walk(rgb_folder)):
                    for pic_name in sorted(pic_names):
                        if "timestamps" not in pic_name and \
                                ".DS_Store" not in pic_name:
                            path = os.path.join(rgb_folder, pic_name)
                            img_path.append(path)

                # append this list into self.images_all_sequence
                self.img_all_sequence.append(img_path)
                # check how many samples in this sequence and
                # add it cumulatively
                cumulative_size = \
                    cumulative_size + len(img_path)-(self.time_step-1)
                self.cumulative_size.append(cumulative_size)

                # read the steering,throttle, brake data of each sequence,
                steering_path = os.path.join(subfolder,
                                             j,
                                             "vehicle_data/steering.txt")
                throttle_path = os.path.join(subfolder,
                                             j,
                                             "vehicle_data/throttle.txt")
                brake_path = os.path.join(subfolder,
                                          j,
                                          "vehicle_data/brake.txt")
                steering_raw = pd.read_csv(steering_path,
                                           header=None)[0].values.tolist()
                throttle_raw = pd.read_csv(throttle_path,
                                           header=None)[0].values.tolist()
                brake_raw = pd.read_csv(brake_path,
                                        header=None)[0].values.tolist()

                # take one sample every 40 samples,
                for number, item in enumerate(steering_raw):
                    if number % 40 == 0:
                        steering.append(item)

                for number, item in enumerate(throttle_raw):
                    if number % 40 == 0:
                        throttle.append(item)

                for number, item in enumerate(brake_raw):
                    if number % 40 == 0:
                        brake.append(item)

                # after taking the actions from this sequence,
                # emerge these actions and append this list to
                # self.action_all_sequence
                action = np.array([steering, throttle, brake]).T
                counter_output = counter_output + action.shape[0]

                self.action_all_sequence.append(action)

        # after iterating over data, get self.action_all_sequences,
        # and self.img_all_sequences
        print(self.cumulative_size)
        print(len(self.cumulative_size))

        # check if sequence number of imgs match the sequence number
        # of actions
        assert len(self.action_all_sequence) == len(self.img_all_sequence), \
            "the sequences of images and the sequences of actions dont match"

        # check if the number of all pics matches the number of all actions
        assert \
            cumulative_size + (self.time_step-1)*len(self.cumulative_size) == counter_output, \
            "the number of states don't match the number of actions"
        self.n_data = counter_output  # how many samples
        self.length = cumulative_size

        self.dtype = np.uint8  # more often used to store the pic.
        self.transform = transform
        print(f"number of data: {self.n_data}")
        print(f"observation type is {self.dtype}")

    def __len__(self):
        """Return the length of the dataset."""
        return self.length
        # return self.n_data - (self.time_step-1) * len(self.cumulative_size)

    def __getitem__(self, idx):
        """Take state-action pairs from the dataset."""
        s_o = []
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index "
                                 "should not exceed dataset length")
            idx = len(self) + idx

        if idx < len(self):  # len(self) is calling __len__(self)
            dataset_idx = bisect.bisect_right(self.cumulative_size, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_size[dataset_idx - 1]

            # take the pic from [sample_idx, sample_idx+self.timestep)
            path = self.img_all_sequence[dataset_idx][sample_idx:sample_idx + self.time_step]

            # a list contains all the keys of the images
            for j in range(self.time_step):
                image = np.asarray(Image.open(path[j]), dtype=self.dtype)
                # after getting picture as an array,
                # 50% to add the shadow, only in train mode
                if np.random.rand() > 0.5 and self.mode == 'train':
                    # adding the shadow, pic augmentation
                    image = _augment_single_image_with_shadow(image, 0.5)
                    # gamma from (1/1.5, 1.5), need an array of type uint8

                if self.transform is not None:
                    image = self.transform(image)
                s_o.append(image)

            # take corresponding action_pairs (sampels, action_pairs =3)
            a_c = (self.action_all_sequence[dataset_idx][sample_idx: sample_idx + self.time_step, :]).astype(np.float32)
            a_c = torch.tensor(a_c)
            # after having all tensors in the so list,
            # change the list to the tensor
            s_o = torch.stack(s_o)

        # return the transformed array (image) and
        # the corresponding label (action)
        return s_o, a_c


class MarkovProcessCNNPng(torch.utils.data.Dataset):
    """This class for CNN training, read image in original format."""

    def __init__(self, dir_name, transform=None, mode='train'):
        """Initialize the object."""
        self.mode = mode
        self.action = []
        # read the observation
        # store all the paths to each image, the steering,throttle, brake
        self.images, steering, throttle, brake = [], [], [], []

        data_town = os.listdir(dir_name)
        data_town = list(filter(lambda a: '.DS_Store' not in a, data_town))
        for i in range(0, len(data_town)):
            name_town = data_town[i]
            subfolder = os.path.join(dir_name, name_town)
            sequence = os.listdir(subfolder)
            sequence = list(filter(lambda a: '.DS_Store' not in a, sequence))
            for j in sequence:
                rgb_folder = os.path.join(subfolder, j, PIC_FOLDER)
                for _, _, pic_names in sorted(os.walk(rgb_folder)):
                    for pic_name in sorted(pic_names):
                        if "timestamps" not in pic_name and \
                                ".DS_Store" not in pic_name:
                            # the rgb_image path
                            path = os.path.join(rgb_folder, pic_name)
                            self.images.append(path)

                # read the steering,throttle, brake data of each sequence,
                steering_path = os.path.join(subfolder,
                                             j,
                                             "vehicle_data/steering.txt")
                throttle_path = os.path.join(subfolder,
                                             j,
                                             "vehicle_data/throttle.txt")
                brake_path = os.path.join(subfolder,
                                          j,
                                          "vehicle_data/brake.txt")
                steering_raw = pd.read_csv(steering_path,
                                           header=None)[0].values.tolist()
                throttle_raw = pd.read_csv(throttle_path,
                                           header=None)[0].values.tolist()
                brake_raw = pd.read_csv(brake_path,
                                        header=None)[0].values.tolist()

                # take one sample every 40 samples,
                for number, item in enumerate(steering_raw):
                    if number % 40 == 0:
                        steering.append(item)

                for number, item in enumerate(throttle_raw):
                    if number % 40 == 0:
                        throttle.append(item)

                for number, item in enumerate(brake_raw):
                    if number % 40 == 0:
                        brake.append(item)
        self.action = [steering, throttle, brake]

        # convert the list of state and list of action to array
        self.action = np.array(self.action).T
        print(self.action.shape)
        # check if the number of pics matches the number of actions
        print(len(self.images))
        print(self.action.shape[0])
        assert len(self.images) == self.action.shape[0],\
            "the number of states dont match the number of actions"
        self.n_data = self.action.shape[0]  # how many samples

        self.dtype = np.uint8  # more often used to store the pic.
        self.transform = transform
        print(f"number of data: {self.n_data}")
        print(f"observation type is {self.dtype}")

    def __len__(self):
        """Return the length of the dataset."""
        return self.n_data   # return the number of total training samples

    def __getitem__(self, idx):
        """Take state-action pairs from the dataset."""
        if idx < self.n_data:

            # list contain all the paths of the images
            path = self.images[idx]

            image_array = np.asarray(Image.open(path))
            s_o = image_array.astype(self.dtype)
            a_c = self.action[idx].astype(np.float32)
            if np.random.rand() > 0.5 and self.mode == 'train':
                # adding the shadow, pic augmentation, only in train mode
                # gamma from (1/1.5, 1.5), need an array of type uint8
                s_o = _augment_single_image_with_shadow(s_o, 0.5)

            if self.transform is not None:
                s_o = self.transform(s_o)

        return s_o, a_c
