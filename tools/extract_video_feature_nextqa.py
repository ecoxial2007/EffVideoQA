import os
import glob
import clip
import torch
import skvideo.io
from PIL import Image
import h5py

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)

def extract_feature(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features



if __name__ == '__main__':
    video_root = 'path_to_your_source_data'
    feature_root = 'path_to_your_feature'
    video_paths = glob.glob(os.path.join(video_root, '*', '*', '*', '*.mp4'))
    num_clips = 16
    frame_per_clip = 4

    for video_path in video_paths:
        video_data = skvideo.io.vread(video_path)
        _, name = os.path.split(video_path)
        metadata = skvideo.io.ffprobe(video_path)

        # frame_per_second_str = metadata["video"]['@r_frame_rate'].split('/')
        # frame_per_second = int(frame_per_second_str[0])/int(frame_per_second_str[1])

        feature_path = os.path.join(feature_root, name.replace('.mp4', '.h5'))
        _, name = os.path.split(video_path)

        if os.path.exists(feature_path): continue
        total_frame = video_data.shape[0]
        sample_rate = total_frame / num_clips / frame_per_clip
        image_features = []
        for mark_id in range(num_clips*frame_per_clip):
            fid = int(mark_id*sample_rate)
            frame = video_data[fid]
            image = Image.fromarray(frame)
            image_features.append(extract_feature(image))

        image_features = torch.cat(image_features, dim=0).cpu().numpy()
        datafile = h5py.File(feature_path, 'w')
        datafile.create_dataset('video_features', data=image_features)
        datafile.close()
        print('Finish extract', video_path)





