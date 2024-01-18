import numpy as np
import torch
import torch.utils.data
import os
import scipy.io as sio


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, length=1024, mode='train'):
        dataset_folder = ["12k Drive End Bearing Fault Data",
                          "12k Fan End Bearing Fault Data",
                          "48k Drive End Bearing Fault Data",
                          "Normal Baseline Data"]
        normal_name = ["97.mat"]
        fault_dir = {1: "105.mat", 2: "118.mat", 3: "130.mat",
                     4: "169.mat", 5: "185.mat", 6: "197.mat",
                     7: "209.mat", 8: "222.mat", 9: "234.mat"}
        self.axis = ["_DE_time", "_FE_time", "_BA_time"]
        self.root = root
        self.length = length

        self.normal_path = os.path.join(self.root, dataset_folder[3], normal_name[0])
        data, labels = self.read_data(self.normal_path, 0)
        for i in fault_dir.keys():
            path = os.path.join(self.root, dataset_folder[0], fault_dir[i])
            data1, labels1 = self.read_data(path, i)
            data += data1
            labels += labels1

        # Shuffle the data and labels
        index = np.arange(len(data))
        np.random.shuffle(index)
        self.data = np.array(data)[index]
        self.labels = np.array(labels)[index]

        if mode == 'train':
            self.data = self.data[:int(len(self.data) * 0.6)]
            self.labels = self.labels[:int(len(self.labels) * 0.6)]
        elif mode == 'val':
            self.data = self.data[int(len(self.data) * 0.6):int(len(self.data) * 0.8)]
            self.labels = self.labels[int(len(self.labels) * 0.6):int(len(self.labels) * 0.8)]
        else:
            self.data = self.data[int(len(self.data) * 0.8):]
            self.labels = self.labels[int(len(self.labels) * 0.8):]

    def read_data(self, path, label):
        id = path.split(".")[-2].split(os.sep)[-1]
        if eval(id) < 100:
            mat_name = "X0" + id + self.axis[0]
        else:
            mat_name = "X" + id + self.axis[0]
        full_data = sio.loadmat(path)[mat_name]

        data, lab = [], []
        start, end = 0, self.length
        while end <= full_data.shape[0]:
            data.append(full_data[start:end])
            lab.append(label)
            start += int(self.length / 2)
            end += int(self.length / 2)
        return data, lab

    def __getitem__(self, index):
        data = self.data[index].reshape(1, self.length)
        label = self.labels[index]
        return torch.from_numpy(data).float(), torch.tensor(label).long()

    def __len__(self):
        return len(self.data)
