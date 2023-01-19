python3 src/trainval.py --use_text_query --use_text_cands --n_frames 16  --n_cands 5\
        --dataset 'nextqa_mc' --data_path './data/Annotation' --feature_path '/home/liangx/Data/NeXt-QA'\
        --batch_size 512 --method 'prob' --tao 0.04

python3 src/test.py --use_text_query --use_text_cands --n_frames 64 --n_cands 5\
        --dataset 'nextqa_mc' --split 'test' --data_path './data/Annotation' --feature_path './path_to_your_Data/NeXt-QA'\
        --checkpoint './checkpoints/nextqa_mc/ckpt_0.5738590955734253.pth'\
        --batch_size 512 --method 'prob'

