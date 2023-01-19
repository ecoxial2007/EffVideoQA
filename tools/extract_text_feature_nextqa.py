import glob
import json
import os
import clip
import torch
import h5py
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)


if __name__ == '__main__':
    qa_root = 'path_to_your_feature'
    datafile = h5py.File(os.path.join(qa_root, 'text_features.h5'), 'w')

    splits = ['train', 'val', 'test']

    all_text_features = []
    answers = []
    for split in splits:
        with open(f'../data/Annotation/nextqa/next_{split}_qa.json', 'r') as ff:
            lines = json.load(ff)

        for f in lines:
            print(f)
            qid = f['qid']
            question = f['question']
            answer = f['answer']
            answers.append(int(answer))
            video_id = f['video_id']
            candidates = f['options']
            text_candi_inputs = torch.cat([clip.tokenize(f"{c}") for c in candidates]).to(device)
            text_quesion_inputs = clip.tokenize(question).to(device)
            text_inputs = torch.cat([text_quesion_inputs, text_candi_inputs], dim=0)
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                all_text_features.append(text_features)


    all_text_features = torch.stack(all_text_features,dim=0).cpu().numpy()
    print(all_text_features.shape, np.array(answers).shape)
    datafile.create_dataset('all_text_features', data=all_text_features)
    datafile.create_dataset('all_answer', data=answers)






