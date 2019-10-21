import pickle
import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from tqdm import tqdm
from argparse import *


class TrajectoryDataset(Dataset):
    def __init__(self, raw_data, train_length, predict_length, keys):
        self.data = raw_data
        self.window_size = train_length + predict_length
        self.train_length = train_length
        self.predict_length = predict_length
        self.keys = keys

    def __getitem__(self, index):
        for key in self.keys:
            if index >= len(self.data[key]) - self.window_size + 1:
                index -= len(self.data[key]) - self.window_size + 1
            else:
                data = self.data[key][index : index + self.window_size]
                # print(data[self.predict_length :])
                train_data = data[0: self.train_length]
                predict_data = data[self.train_length:]
                return torch.from_numpy(train_data).type(torch.float32), torch.from_numpy(predict_data).type(torch.float32)

    def __len__(self):
        index = 0
        for key in self.keys:
            index += len(self.data[key]) - self.window_size + 1
        return index


class NewDataLoader():
    def __init__(self, dataset, batch_size=20):
        self.dataset = dataset
        # Store the arguments
        self.batch_size = batch_size
        # Validation arguments
        # self.val_fraction = 0.2
        # Increment the counter with the number of sequences in the current dataset
        # counter = int(len(all_frame_data) / (self.seq_length))
        # valid_counter += int(len(valid_frame_data) / (self.seq_length))
        # Calculate the number of batches
        self.num_batches = int(len(dataset) / self.batch_size)
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0

    def next_batch(self):
        # Frame data
        frame_batch = []

        for i in range(self.dataset_pointer, self.dataset_pointer + self.batch_size):
            sequence = []  # status for all people in seq_length
            for seq in self.dataset[i]:  # len = 2
                for moment in seq:  # len = 30 | 10
                    slice = []  # status for all people at one moment
                    j = 0
                    while j < len(moment):  # len = 44
                        pid = j / 2 + 1
                        pos = [pid, moment[j].item(), moment[j + 1].item()]  # status for one person at one moment
                        slice.append(pos)
                        j += 2
                    sequence.append(np.array(slice))
            # sequence.append(np.array([[0]*3]*22))
            frame_batch.append(sequence)

        self.dataset_pointer += self.batch_size
        return frame_batch

    def reset_batch_pointer(self):
        '''
        Reset pointer
        '''
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0


# class DataLoader():
#
#     def __init__(self, batch_size=20, seq_length=40, datasets=[0], forcePreProcess=False, infer=False):
#         '''
#         Initialiser function for the DataLoader class
#         params:
#         batch_size : Size of the mini-batch
#         seq_length : Sequence length to be considered
#         datasets : The indices of the datasets to use
#         forcePreProcess : Flag to forcefully preprocess the data again from csv files
#         '''
#         # List of data directories where raw data resides
#         self.data_dirs = ['./data/jnli/train']
#         self.used_data_dirs = [self.data_dirs[x] for x in datasets]
#         self.test_data_dirs = [self.data_dirs[x] for x in range(1) if x not in datasets]
#         self.infer = infer
#
#         # Number of datasets
#         self.numDatasets = len(self.data_dirs)
#
#         # Data directory where the pre-processed pickle file resides
#         self.data_dir = './data'
#
#         # Store the arguments
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#
#         # Validation arguments
#         self.val_fraction = 0.2
#
#         # Define the path in which the process data would be stored
#         data_file = os.path.join(self.data_dir + '/jnli', "trajectories.cpkl")
#
#         # If the file doesn't exist or forcePreProcess is true
#         if not (os.path.exists(data_file)) or forcePreProcess:
#             print("Creating pre-processed data from raw data")
#             # Preprocess the data from the csv files of the datasets
#             # Note that this data is processed in frames
#             self.frame_preprocess(self.used_data_dirs, data_file)
#
#         # Load the processed data from the pickle file
#         self.load_preprocessed(data_file)
#         # Reset all the data pointers of the dataloader object
#         self.reset_batch_pointer(valid=False)
#         self.reset_batch_pointer(valid=True)
#
#     def frame_preprocess(self, data_dirs, data_file):
#         '''
#         Function that will pre-process the pixel_pos.csv files of each dataset
#         into data with occupancy grid that can be used
#         params:
#         data_dirs : List of directories where raw data resides
#         data_file : The file into which all the pre-processed data needs to be stored
#         '''
#         # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
#         # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
#         # containing pedID, x, y
#         all_frame_data = []
#         # Validation frame data
#         valid_frame_data = []
#         # frameList_data would be a list of lists corresponding to each dataset
#         # Each list would contain the frameIds of all the frames in the dataset
#         frameList_data = []
#         # numPeds_data would be a list of lists corresponding to each dataset
#         # Ech list would contain the number of pedestrians in each frame in the dataset
#         numPeds_data = []
#         # Index of the current dataset
#         dataset_index = 0
#         print('lol')
#
#         # For each dataset
#         for ind_directory, directory in enumerate(data_dirs):
#             # define path of the csv file of the current dataset
#             # file_path = os.path.join(directory, 'pixel_pos.csv')
#             file_path = os.path.join(directory, 'pixel_pos_interpolate100.csv')
#
#             # Load the data from the csv file
#             data = np.genfromtxt(file_path, delimiter=',')
#             print(len(data))
#
#             # Frame IDs of the frames in the current dataset
#             frameList = np.unique(data[0, :]).tolist()
#             numFrames = len(frameList)
#             print(numFrames)
#
#             # Add the list of frameIDs to the frameList_data
#             frameList_data.append(frameList)
#             # Initialize the list of numPeds for the current dataset
#             numPeds_data.append([])
#             # Initialize the list of numpy arrays for the current dataset
#             all_frame_data.append([])
#             # Initialize the list of numpy arrays for the current dataset
#             valid_frame_data.append([])
#
#             # if directory == './data/eth/univ':
#             #    skip = 6
#             # else:
#             #    skip = 10
#             # skip = 3
#
#             skip = 10
#
#             for ind, frame in enumerate(frameList):
#
#                 ## NOTE CHANGE
#                 if ind % skip != 0:
#                     # Skip every n frames
#                     continue
#
#                 # Extract all pedestrians in current frame
#                 pedsInFrame = data[:, data[0, :] == frame]
#
#                 # Extract peds list
#                 pedsList = pedsInFrame[1, :].tolist()
#
#                 # Add number of peds in the current frame to the stored data
#                 numPeds_data[dataset_index].append(len(pedsList))
#
#                 # Initialize the row of the numpy array
#                 pedsWithPos = []
#
#                 # For each ped in the current frame
#                 for ped in pedsList:
#                     # Extract their x and y positions
#                     current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
#                     current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]
#
#                     # Add their pedID, x, y to the row of the numpy array
#                     pedsWithPos.append([ped, current_x, current_y])
#
#                 if (ind > numFrames * self.val_fraction) or (self.infer):
#                     # At inference time, no validation data
#                     # Add the details of all the peds in the current frame to all_frame_data
#                     all_frame_data[dataset_index].append(np.array(pedsWithPos))
#                 else:
#                     valid_frame_data[dataset_index].append(np.array(pedsWithPos))
#
#             dataset_index += 1
#
#         # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
#         f = open(data_file, "wb")
#         pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data), f, protocol=2)
#         f.close()
#
#     def load_preprocessed(self, data_file):
#         '''
#         Function to load the pre-processed data into the DataLoader object
#         params:
#         data_file : the path to the pickled data file
#         '''
#         # Load data from the pickled file
#         f = open(data_file, 'rb')
#         self.raw_data = pickle.load(f)
#         print(len(self.raw_data))
#         print(len(self.raw_data[0]))
#         f.close()
#         # Get all the data from the pickle file
#         self.data = self.raw_data[0]
#         self.frameList = self.raw_data[1]
#         self.numPedsList = self.raw_data[2]
#         self.valid_data = self.raw_data[3]
#         counter = 0
#         valid_counter = 0
#
#         # For each dataset
#         for dataset in range(len(self.data)):
#             # get the frame data for the current dataset
#             all_frame_data = self.data[dataset]
#             valid_frame_data = self.valid_data[dataset]
#             print('Training data from dataset {} : {}'.format(dataset, len(all_frame_data)))
#             print('Validation data from dataset {} : {}'.format(dataset, len(valid_frame_data)))
#             # Increment the counter with the number of sequences in the current dataset
#             counter += int(len(all_frame_data) / (self.seq_length))
#             valid_counter += int(len(valid_frame_data) / (self.seq_length))
#
#         # Calculate the number of batches
#         print(counter)
#         print(valid_counter)
#         self.num_batches = int(counter / self.batch_size)
#         self.valid_num_batches = int(valid_counter / self.batch_size)
#         print('Total number of training batches: {}'.format(self.num_batches * 2))
#         print('Total number of validation batches: {}'.format(self.valid_num_batches))
#         # On an average, we need twice the number of batches to cover the data
#         # due to randomization introduced
#         self.num_batches = self.num_batches * 2
#         # self.valid_num_batches = self.valid_num_batches * 2
#
#     def next_batch(self, randomUpdate=True):
#         '''
#         Function to get the next batch of points
#         '''
#         # Source data
#         x_batch = []
#         # Target data
#         y_batch = []
#         # Frame data
#         frame_batch = []
#         # Dataset data
#         d = []
#         # Iteration index
#         i = 0
#         while i < self.batch_size:
#             # Extract the frame data of the current dataset
#             frame_data = self.data[self.dataset_pointer]
#             frame_ids = self.frameList[self.dataset_pointer]
#             # Get the frame pointer for the current dataset
#             idx = self.frame_pointer
#             # While there is still seq_length number of frames left in the current dataset
#             if idx + self.seq_length < len(frame_data):
#                 # All the data in this sequence
#                 # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
#                 seq_source_frame_data = frame_data[idx:idx + self.seq_length]
#                 seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]
#                 seq_frame_ids = frame_ids[idx:idx + self.seq_length]
#
#                 # Number of unique peds in this sequence of frames
#                 x_batch.append(seq_source_frame_data)
#                 y_batch.append(seq_target_frame_data)
#                 frame_batch.append(seq_frame_ids)
#
#                 # advance the frame pointer to a random point
#                 if randomUpdate:
#                     self.frame_pointer += random.randint(1, self.seq_length)
#                 else:
#                     self.frame_pointer += self.seq_length
#
#                 d.append(self.dataset_pointer)
#                 i += 1
#
#             else:
#                 # Not enough frames left
#                 # Increment the dataset pointer and set the frame_pointer to zero
#                 self.tick_batch_pointer(valid=False)
#
#         return x_batch, y_batch, frame_batch, d
#
#     def next_valid_batch(self, randomUpdate=True):
#         '''
#         Function to get the next Validation batch of points
#         '''
#         # Source data
#         x_batch = []
#         # Target data
#         y_batch = []
#         # Dataset data
#         d = []
#         # Iteration index
#         i = 0
#         while i < self.batch_size:
#             # Extract the frame data of the current dataset
#             frame_data = self.valid_data[self.valid_dataset_pointer]
#             # Get the frame pointer for the current dataset
#             idx = self.valid_frame_pointer
#             # While there is still seq_length number of frames left in the current dataset
#             if idx + self.seq_length < len(frame_data):
#                 # All the data in this sequence
#                 # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
#                 seq_source_frame_data = frame_data[idx:idx + self.seq_length]
#                 seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]
#
#                 # Number of unique peds in this sequence of frames
#                 x_batch.append(seq_source_frame_data)
#                 y_batch.append(seq_target_frame_data)
#
#                 # advance the frame pointer to a random point
#                 if randomUpdate:
#                     self.valid_frame_pointer += random.randint(1, self.seq_length)
#                 else:
#                     self.valid_frame_pointer += self.seq_length
#
#                 d.append(self.valid_dataset_pointer)
#                 i += 1
#
#             else:
#                 # Not enough frames left
#                 # Increment the dataset pointer and set the frame_pointer to zero
#                 self.tick_batch_pointer(valid=True)
#
#         return x_batch, y_batch, d
#
#     def tick_batch_pointer(self, valid=False):
#         '''
#         Advance the dataset pointer
#         '''
#         if not valid:
#             # Go to the next dataset
#             self.dataset_pointer += 1
#             # Set the frame pointer to zero for the current dataset
#             self.frame_pointer = 0
#             # If all datasets are done, then go to the first one again
#             if self.dataset_pointer >= len(self.data):
#                 self.dataset_pointer = 0
#         else:
#             # Go to the next dataset
#             self.valid_dataset_pointer += 1
#             # Set the frame pointer to zero for the current dataset
#             self.valid_frame_pointer = 0
#             # If all datasets are done, then go to the first one again
#             if self.valid_dataset_pointer >= len(self.valid_data):
#                 self.valid_dataset_pointer = 0
#
#     def reset_batch_pointer(self, valid=False):
#         '''
#         Reset all pointers
#         '''
#         if not valid:
#             # Go to the first frame of the first dataset
#             self.dataset_pointer = 0
#             self.frame_pointer = 0
#         else:
#             self.valid_dataset_pointer = 0
#             self.valid_frame_pointer = 0
#
#
def transfer_data():
    parser = ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='./data_load2.npz')
    parser.add_argument('--train_length', type=int, default=30)
    parser.add_argument('--predict_length', type=int, default=10)
    args = parser.parse_args()
    data_path = args.dataPath
    train_length = args.train_length
    predict_length = args.predict_length
    data = np.load(data_path)
    keys = list(data.keys())
    keys_train = keys[0: -3]
    keys_eval = keys[-2:]
    dataset_train = TrajectoryDataset(data, train_length, predict_length, keys_train)

    dataloader = NewDataLoader(dataset_train)
    batch = dataloader.next_batch()
    print(batch[0][0][0])
    print(len(batch[0][0][0]))
    print(len(batch[0][0]))
    print(len(batch[0]))
    print(len(batch))
    batch = dataloader.next_batch()
    print(batch[0][0][0])
    raise KeyError

    dataset_eval = TrajectoryDataset(data, train_length, predict_length, keys_eval)
    f_path = './data/jnli'
    if not os.path.exists(f_path):
        os.mkdir(f_path)
    if not os.path.exists(f_path + '/train'):
        os.mkdir(f_path + '/train')
    if not os.path.exists(f_path + '/eval'):
        os.mkdir(f_path + '/eval')

    def save_data(datasets, name):
        f = open(f_path + '/' + name + '/pixel_pos_interpolate100.csv', 'w', encoding='utf8')
        t = 1
        time_line = ''
        pid_line = ''
        x_line = ''
        y_line = ''
        i = 0
        for dataset in tqdm(datasets):
            i += 1
            if i > 100:
                break
            try:
                for data in dataset:
                    for moment in data:
                        people = [moment[i:i+2] for i in range(0, len(moment), 2)]
                        for pid, pos in enumerate(people):
                            assert len(pos) == 2
                            time_line += str(t) + ','
                            pid_line += str(pid + 1) + ','
                            x_line += str(pos[0].item()) + ','
                            y_line += str(pos[1].item()) + ','
                        t += 1
            except:
                break
        f.write(time_line[:-1] + '\n' + pid_line[:-1] + '\n' + x_line[:-1] + '\n' + y_line[:-1])
        f.close()

    save_data(dataset_train, 'train')
    # save_data(dataset_eval, 'eval')
    print(len(dataset_train))
    print(len(dataset_train[0]))
    print(len(dataset_train[-1][0]))
    print(len(dataset_train[0][0][0]))
    # print(len(dataset_train))
    # print(len(dataset_train[0]))
    # print(len(dataset_train[-1][0]))
    # print(len(dataset_train[0][0][0]))


if __name__ == '__main__':
    transfer_data()
    # dataloader = DataLoader()
