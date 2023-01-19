"""
Simple video-language dataset class file, for illustration purposes. See comments below for relevant highlights.
"""
import os
import json
import h5py
import numpy as np
import random
import torch
from torch.utils import data



class VideoLanguageDataset(data.Dataset):
    def __init__(self, args, split="train"):
        super().__init__()
        self.data_path = args.data_path
        self.feature_path = args.feature_path
        self.split = split
        self.get_text_query = args.use_text_query
        self.get_text_cands = args.use_text_cands
        self.n_frames = args.n_frames
        self.visible = args.visible
        self.mixup = args.mixup
        self.shuffle = args.shuffle

        with open(os.path.join(self.data_path, 'nextqa', f'next_{split}_qa.json'), 'r') as jf:
            self.metadata = json.load(jf)

        self.text_features = h5py.File(os.path.join(self.feature_path, 'text_features_clip.h5'), 'r')['all_text_features']
        video_ids = []
        for f in self.metadata:
            video_ids.append(f['video_id'])
        self.video_ids = list(set(video_ids))
        print(self.split, len(self.metadata))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Assuming torch files for each of the features for simplicity; adjust to fit your extracted features.
        (e.g. numpy, hdf5, json, etc.)
        """

        f = self.metadata[index]
        qid = int(f['qid'])
        video_id = int(f['video_id'])
        question = f['question']
        typee = f['type']
        labels_id = torch.tensor(int(f['answer']), dtype=torch.long)
        options = f['options']

        video_feature_path = os.path.join(self.feature_path, 'NeXt-clip-features', f'{video_id}.h5')
        video_features = h5py.File(video_feature_path, 'r')['video_features']
        video_features = torch.tensor(np.array(video_features),
                                      dtype=torch.float32)  # (L_video, D_in); L_video >> L

        frame_idxs_gt = torch.randperm(len(video_features))[:self.n_frames]
        if not self.shuffle:
            frame_idxs_gt = torch.sort(frame_idxs_gt).values

        video_features_sampled = video_features[frame_idxs_gt]  # (L, D_in)
        if self.mixup and self.split == 'train':
            select_frame = random.randint(0, 4)
            alpha = select_frame / self.n_frames
            select_idx = torch.randperm(self.n_frames)[:select_frame]
            for replace_id in select_idx:
                select_video_id = random.choice(self.video_ids)
                select_video_feature_path = os.path.join(self.feature_path, 'NeXt-clip-features',
                                                         f'{select_video_id}.h5')
                select_video_features = h5py.File(select_video_feature_path, 'r')['video_features']
                select_video_features = torch.tensor(np.array(select_video_features), dtype=torch.float32)
                video_features_sampled[replace_id] = random.choice(select_video_features)
        else:
            alpha = 0

        text_query_features = torch.tensor(self.text_features[qid][0],
                                           dtype=torch.float32) if self.get_text_query else []
        text_cands_features = torch.tensor(self.text_features[qid][1:],
                                           dtype=torch.float32) if self.get_text_cands else []

        if self.visible:
            return (video_features_sampled, frame_idxs_gt, text_query_features, text_cands_features, labels_id), (video_id, question, options, typee)
        else:
            return video_features_sampled, frame_idxs_gt, text_query_features, text_cands_features, labels_id, alpha




