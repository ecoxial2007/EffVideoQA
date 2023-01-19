# Efficient Video Question Answering
This is the implementation of our paper, all features and weights will be released on github. 
You can also extract video and text features yourself according to our code and documentation.

## Environment
This code is tested with:
- Ubuntu 20.04
- PyTorch >= 1.8
- CUDA >= 10.1

```
# create your virtual environment
conda create --name effvqa python=3.7
conda activate atp

# dependencies
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch
conda install pandas

# optional (for feature extraction); see also tools/*.py
pip install git+https://github.com/openai/CLIP.git
```

## Dataset
- Annotation: check `./data/Annotation`
- Source data:
  - Next-QA: https://xdshang.github.io/docs/vidor.html
  - MSR-VTT & MSVD: https://github.com/xudejing/video-question-answering
  - ActivityNet-QA: https://github.com/MILVLG/activitynet-qa
  - TGIF: https://github.com/YunseokJANG/tgif-qa

## Feature extraction
Please refer to `./tools/extract_video_feature_nextqa.py` and `./tools/extract_text_feature_nextqa.py`

## Train & Val & Test
Check `trainval.sh`
```
python3 src/trainval.py --use_text_query --use_text_cands --n_frames 16  --n_cands 5\
        --dataset 'nextqa_mc' --data_path './data/Annotation' --feature_path './path_to_your_Data/NeXt-QA'\
        --batch_size 512 --method 'prob' --tao 0.04

python3 src/test.py --use_text_query --use_text_cands --n_frames 64 --n_cands 5\
        --dataset 'nextqa_mc' --split 'test' --data_path './data/Annotation' --feature_path './path_to_your_Data/NeXt-QA'\
        --checkpoint './checkpoints/nextqa_mc/ckpt_0.5745.pth'\
        --batch_size 512 --method 'prob'
```

### LICENSE / Contact

We release this repo under the open [MIT License](LICENSE). 

# Acknowledgements
We reference the excellent repos of NeXT-QA, ATP, CLIP, in addition to other specific repos to the datasets/baselines we examined (see paper). If you build on this work, please be sure to cite these works/repos as well.
