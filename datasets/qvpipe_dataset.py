import os
import random
import glob
import json
import numpy as np
import random
#np.random.seed(1)
#random.seed(1)
import math
import cv2
import pdb
import yaml

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
#torch.manual_seed(17)

def load_data(root, group):
    ann_file = os.path.join(root, 'qv_pipe_train.json')
    train_file = os.path.join(root, 'train_keys.json')
    val_file = os.path.join(root, 'val_keys.json')
    with open(ann_file, 'r') as fp:
        data = json.load(fp)
    with open(train_file, 'r') as fp:
        train_keys = json.load(fp)
    with open(val_file, 'r') as fp:
        val_keys = json.load(fp)
    test_keys = []
    all_files = sorted(glob.glob(os.path.join(root, 'track1_raw_video') + '/*.mp4'))
    for key in all_files:
        fname = os.path.basename(key)
        if fname not in train_keys and fname not in val_keys:
            test_keys.append(fname)

    if group == 5: # use full training set for training
        train_keys.extend(val_keys)
    else: # use 6000 samples for training, 399 for validation
        train_keys.extend(val_keys[:2000])

    val_keys = val_keys[2000:]

    if group == 1: # This is used for 2-3 epochs; only train on class 6, 8, 11, 13, 14, 15
        to_delete_cls = [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 16]
        for del_cls in to_delete_cls:
            train_keys = [key for key in train_keys if del_cls not in data[key]]
    elif group == 2: #  not used
        to_delete_cls = [1, 4, 5, 7, 9, 10, 16]
        for del_cls in to_delete_cls:
            train_keys = [key for key in train_keys if del_cls not in data[key]]
    elif group == 3 or group == 5:
        pass
    else: # group 4 -> this is the default: remove ~1000 samples from class 0
        train_keys[:-2000] = [key for key in train_keys[:-2000] if 0 not in data[key]]

    return data, train_keys, val_keys, test_keys


def clip_loader_test(path, num_key_frames=5):
    cap = cv2.VideoCapture(path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = int(num_frames)
    all_key_frames = np.array(list(range(num_frames)))
    if num_frames == 0:
        print('this path has zero frames ', path)
    if num_frames < num_key_frames:
        key_frames = all_key_frames[np.random.choice(len(all_key_frames), size=num_key_frames, replace=True)]
    else:
        key_frames = all_key_frames[np.random.choice(len(all_key_frames), size=num_key_frames, replace=False)]
    key_frames = sorted(key_frames)
    frame_id = 0
    frames = []
    for selected_frame in key_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        success, frame = cap.read()
        frame = frame[:, :, ::-1].copy()
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames

class QVPipeDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_type='train', group=4, num_key_frames=5, transform = None, evaluation=False):
        self.root = root
        self.data, train_keys, val_keys, test_keys = load_data(root, group)
        if dataset_type == 'train':
            self.keys = train_keys
        elif dataset_type == 'val':
            self.keys = val_keys
        else:
            self.keys = test_keys

        if len(self.keys) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.evaluation = evaluation
        self.transform = transform
        self.dataset_type = dataset_type
        self.num_classes = 17
        self.group = group
        self.num_key_frames = num_key_frames

    def __getitem__(self, index):

        video_name = self.keys[index]

        video_file = os.path.join(self.root, 'track1_raw_video', video_name)

        if self.dataset_type != 'test':
            img = clip_loader_test(video_file, self.num_key_frames)
            if self.transform is not None:
                clip = torch.from_numpy(img)
                clip = clip.permute((0, 3, 1, 2)).contiguous()
                clip = clip.to(dtype=torch.get_default_dtype()).div(255)
                clip = self.transform(clip)
        else:
            clips = []
            for idx in range(5):
                img = clip_loader_test(video_file)
                if self.transform is not None:
                    clip = torch.from_numpy(img)
                    clip = clip.permute((0, 3, 1, 2)).contiguous()
                    clip = clip.to(dtype=torch.get_default_dtype()).div(255)
                    clip = self.transform(clip)
                    clips.append(clip)

        if self.dataset_type != 'test':
            # multi-hot encoding
            if img is not None:
                labels = self.data[self.keys[index]]
                labels = torch.tensor(labels)
                labels = labels.unsqueeze(0)
                labels = torch.zeros(labels.size(0), self.num_classes).scatter_(1, labels, 1.)
                labels = labels.squeeze(0)
            else:
                labels = torch.zeros(self.num_classes)
        else:
            # if we're using the validation set, we do have the labels available
            if self.keys[index] in self.data.keys():
                labels = self.data[self.keys[index]]
            else:
                labels = [0]
            labels = torch.tensor(labels)
            labels = labels.unsqueeze(0)
            labels = torch.zeros(labels.size(0), self.num_classes).scatter_(1, labels, 1.)
            labels = labels.squeeze(0)

        if self.dataset_type != 'test':
            return clip, labels, video_name
        else:
            return clips, labels, video_name

    def __len__(self):
        return len(self.keys)


def main():
    ### Only for testing the QVPipeDataset class
    transform = transforms.Compose(
        [
            transforms.Resize(300),
            transforms.RandomCrop(300),
            transforms.RandomAdjustSharpness(1.5),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    dataset = QVPipeDataset('classification_track', dataset_type='train', group=4, transform=transform, evaluation=True)
    import matplotlib.pyplot as plt
    all_ids = np.array(list(range(len(dataset))))
    ids = all_ids[np.random.choice(len(all_ids), size=10, replace=False)]
    for idd in ids:
        clip, labels, name = dataset[idd]
        clip = clip.permute(0, 2,3,1).reshape(5*224, 224, 3).numpy()
        plt.imshow(clip)
        plt.show()

if __name__ == '__main__':
    main()
