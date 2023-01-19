import os
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from model import EffConfig, EfficientModel, downstream_task_forward

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)
    args.log_path = os.path.join("./checkpoints", args.dataset)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    with open(os.path.join(args.log_path, 'log.txt'), 'a') as lf:
        lf.write(message+'\n')

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]
    if replace_empty_with_none:
        batch = [_b if len(_b) > 0 else None for _b in batch]
    return batch

def main(args):
    seed_everything(args.seed)
    # create EfficientModel from model hyperparameters
    config = EffConfig.from_args(args)
    device = torch.device("cuda")
    model = EfficientModel(config, device).to(device)

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


    dset_train = VideoLanguageDataset(args, split="train")
    dset_val = VideoLanguageDataset(args, split="val")
    dldr_train = torch.utils.data.DataLoader(dset_train,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)
    dldr_val   = torch.utils.data.DataLoader(dset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    # create optimizer
    if args.wd > 0.0:
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.wd)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # simple training loop (for illustrative purposes)
    for epoch_i in range(args.epochs):
        # train epoch
        model.train()
        for i, batch in enumerate(dldr_train):
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)


            # refactored the "forward pass" here
            loss, accs, selected_frames, y_pred = downstream_task_forward(model, batch)
            model.zero_grad(set_to_none=True)

            loss.backward()

            # do logging stuff with accs, loss, etc. For example:
            log(f"train: epoch{epoch_i}, iter{i}: loss = {loss.item()}, acc = {accs.mean().item()}")
            if args.grad_clip_val > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_val)
            optim.step()

        # val epoch
        model.eval()
        all_val_accs = []
        for i, batch in enumerate(dldr_val):
            with torch.no_grad():
                batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
                loss, accs, selected_frames, y_pred = downstream_task_forward(model, batch)
                all_val_accs.append(accs)
        overall_acc = torch.cat(all_val_accs).mean().item()
        log(f"val: epoch{epoch_i}: overall_acc = {overall_acc}")

        checkpoint = {
            "epoch": epoch_i,
            "overall_acc": overall_acc,
            "state_dict": model.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.log_path, f"ckpt_{overall_acc}.pth"))
    return 0



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for FrameAggregation example training script.")

    # Training hyperparameters
    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--wd', default=0.0, type=float, help="weight decay")
    parser.add_argument('--epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--grad_clip_val', default=1.0, type=float, help="gradient clip, must be set > 0 to enable")
    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=4, type=int, help="number of dataset workers")
    parser.add_argument('--seed', default=2022, type=int, help="random seed")

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
    parser.add_argument('--tao', default=1.0, type=float, help="see EffConfig")
    args = parser.parse_args()

    main(args)