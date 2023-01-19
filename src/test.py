import os
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from model import EffConfig, EfficientModel, downstream_task_forward

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    try: #if visible
        t_batch, v_batch = batch
    except:
        t_batch = batch
    if set_to_device is not None:
        t_batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in t_batch]
    if replace_empty_with_none:
        t_batch = [_b if len(_b) > 0 else None for _b in t_batch]
    try:
        return t_batch, v_batch
    except:
        return t_batch


def main(args):
    config = EffConfig.from_args(args)
    device = torch.device("cuda")
    model = EfficientModel(config, device).to(device)

    checkpoints = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict=checkpoints, strict=True)

    if args.dataset == 'nextqa_mc':
        from datasets.nextqa import VideoLanguageDataset
    # elif args.dataset == 'msrvtt_oe' or args.dataset == 'msrvtt_mc':
    #     from datasets.msrvtt import VideoLanguageDataset
    # elif args.dataset == 'msvd_oe':
    #     from datasets.msvd import VideoLanguageDataset
    # elif args.dataset == 'tgif_action_mc' or args.dataset == 'tgif_transition_mc' or args.dataset == 'tgif_frameqa_oe':
    #     from datasets.tgif import VideoLanguageDataset
    # elif args.dataset == 'activitynet_oe':
    #     from datasets.activitynet import VideoLanguageDataset
    # elif args.dataset == 'ucf101_oe':
    #     from datasets.ucf101 import VideoLanguageDataset
    # elif args.dataset == 'hmdb51_oe':
    #     from datasets.hmdb51 import VideoLanguageDataset

    dset_test = VideoLanguageDataset(args, split=args.split)
    dldr_test = torch.utils.data.DataLoader(dset_test,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             drop_last=True,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    # val epoch
    model.eval()
    all_test_accs = []
    for i, batch in enumerate(dldr_test):
        with torch.no_grad():
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            loss, accs, selected_frames, y_pred = downstream_task_forward(model, batch)
            all_test_accs.append(accs)
    overall_acc = torch.cat(all_test_accs).mean().item()
    print(overall_acc)
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for FrameAggregation example training script.")

    # Training hyperparameters
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=8, type=int, help="number of dataset workers")

    # Efficient model hyperparameters (for more help/details, see EffConfig)
    parser.add_argument('--n_layers', default=1, type=int, help="see EffConfig")
    parser.add_argument('--n_heads', default=12, type=int, help="see EffConfig")
    parser.add_argument('--enc_dropout', default=0.1, type=float, help="see EffConfig")
    parser.add_argument('--d_input', default=768, type=int, help="see EffConfig")
    parser.add_argument('--method', default='prob', type=str, help="Ablation, our method called prob",
                        choices=['prob', 'temp', 'self', 'mean'])

    # I/O and tools parameters
    parser.add_argument('--data_path', type=str, help='Annotation')
    parser.add_argument('--feature_path', type=str, help='Feature')
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--split', type=str)
    parser.add_argument('--dataset', type=str,
                        choices=['hmdb51_oe', 'ucf101_oe', 'msvd_oe', 'nextqa_mc', 'msrvtt_mc', 'msrvtt_oe',
                                 'tgif_action_mc', 'tgif_transition_mc', 'tgif_frameqa_oe', 'activitynet_oe'])
    parser.add_argument('--n_frames', default=16, type=int, help="number of frames sampled for input; see tools.py")
    parser.add_argument('--use_text_query', action='store_true')
    parser.add_argument('--use_text_cands', action='store_true')
    parser.add_argument('--n_cands', type=int, default=1000)
    parser.add_argument('--mixup', action='store_true') #no use
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--visible', action='store_true') #for check each question predicts answer
    parser.add_argument('--tao', default=0.04, type=float, help="see EffConfig")
    args = parser.parse_args()

    main(args)