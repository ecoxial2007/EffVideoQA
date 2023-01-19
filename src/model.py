"""
Main file containing core EffVQA class code.
"""
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat
from typing import Optional
from utils.attn import LayerNorm, ProbAttention


@dataclass
class EffConfig:
    # Frame Aggregation Encoder params
    method: str = 'prob'
    split: str = 'train'
    n_layers: int = 1
    n_heads: int = 12
    n_frames: int = 16
    enc_dropout: float = 0.1
    use_text_query: bool = True  # at least one use_text_* needs to be true for EffVQA to be multimodal
    use_text_cands: bool = True  # ^ see above. (note: if both are false, EffVQA is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    d_input: int = 768 # size of the input vision-language embeddings
    tao: float = 1.0
    @classmethod
    def from_args(cls, args):
        return cls(method=args.method,
                   n_layers=args.n_layers,
                   n_heads=args.n_heads,
                   enc_dropout=args.enc_dropout,
                   use_text_query=args.use_text_query,
                   use_text_cands=args.use_text_cands,
                   n_cands=args.n_cands,
                   d_input=args.d_input,
                   n_frames=args.n_frames,
                   tao=args.tao,
                   split=args.split
                   )

# Borrow code from:
# https://github.com/lucidrains/CoCa-pytorch
# https://arxiv.org/abs/2205.01917

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None, attn_type='mh'):
        super().__init__()
        if attn_type == 'mh':
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        elif attn_type == 'pb':
            self.attn = ProbAttention(attention_dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context, attn = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + context
        q = q + self.mlp(self.ln_2(q))
        return q

class ProbSparse(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(ProbSparse, self).__init__()
        self.width = width
        self.layers = layers
        self.mh_attn = ResidualCrossAttentionBlock(width, heads, dropout, attn_mask, 'mh')
        self.pb_attn = ResidualCrossAttentionBlock(width, heads, dropout, attn_mask, 'pb')

    def forward(self, q: torch.Tensor, k: torch.Tensor=None, v: torch.Tensor=None):
        q = q.permute((1, 0, 2)).unsqueeze(dim=2)
        q = self.pb_attn(q, q, q)
        q = q.squeeze(dim=2).permute((1, 0, 2))
        q = self.mh_attn(q, k, v)
        return q


class TemporalModeling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModeling, self).__init__()
        self.width = width
        self.layers = layers
        self.mh_attn1 = ResidualCrossAttentionBlock(width, heads, dropout, attn_mask, 'mh')
        self.mh_attn2 = ResidualCrossAttentionBlock(width, heads, dropout, attn_mask, 'mh')

    def forward(self, q: torch.Tensor, k: torch.Tensor=None, v: torch.Tensor=None):
        q = self.mh_attn1(q, q, q)
        q = self.mh_attn2(q, k, v)
        return q

class EfficientModel(nn.Module):
    """
    Takes as input a sequence of image-language encoding and outputs weighted frames over the inputs,
    to help analyze downstream discriminative video-language tasks.
    """

    def __init__(self, config: EffConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.method = config.method
        if self.method == 'temp':
            self.frameaggregation = TemporalModeling(width=config.d_input, layers=config.n_layers,
                                                      heads=config.n_heads,
                                                      dropout=config.enc_dropout)
            self.temporalEmbedding = torch.nn.Embedding(config.n_frames, config.d_input)
            self.initialize_parameters()

        elif self.method == 'self':
            self.frameaggregation = ResidualCrossAttentionBlock(d_model=config.d_input, n_head=config.n_heads,
                                                             dropout=config.enc_dropout, attn_type='mh')
        elif self.method == 'prob':
            self.frameaggregation = ProbSparse(width=config.d_input, layers=config.n_layers,
                                                      heads=config.n_heads,
                                                      dropout=config.enc_dropout)

    def initialize_parameters(self):
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_query: Optional[torch.tensor] = None,
                x_txt_cands: Optional[torch.tensor] = None,
                **kwargs):
        """
        Performs the Frame Aggregation operation on the input embeddings.
        Returns weighted visual embeddings.
        x_vis_seq: torch.tensor of shape (N, L, D_in) with visual embeddings of size D_in
        x_txt_query: torch.tensor of shape (N, D_in) with optional query text embeddings
        x_txt_cands: torch.tensor of shape (N, L_cands, D_in) with optional add'l text embeddings
        """

        if self.config.use_text_query:
            qFeature = x_txt_query.unsqueeze(dim=0)
        else:
            qFeature = 0
        if self.config.use_text_cands:
            aFeature = rearrange(x_txt_cands, 'b t c -> t b c')
        else:
            aFeature = 0

        vFeature = rearrange(x_vis_seq, 'b t c -> t b c')
        if self.method == 'prob':
            vFeature = vFeature
            tFeature = aFeature + qFeature
            vFeature = self.frameaggregation(vFeature, tFeature, tFeature)

        elif self.method == 'temp':
            tempEmbedding = repeat(self.temporalEmbedding(torch.arange(self.config.n_frames).to(self.device)), 't c -> t b c',
                                          b=vFeature.size(1))
            vFeature = vFeature + tempEmbedding.to(self.device)
            tFeature = aFeature + qFeature
            vFeature = self.frameaggregation(vFeature, tFeature, tFeature)

        elif self.method == 'self':
            vFeature = torch.cat([vFeature, qFeature], dim=0)
            vFeature = self.frameaggregation(vFeature, vFeature, vFeature)

        elif self.method == 'mean':
            vFeature = vFeature

        return vFeature.mean(dim=0)




## eps label smooth
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,  reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, target, alpha):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        return reduce_loss(alpha * loss/n + (1 - alpha) * nll, self.reduction)


label_smooth_cross_entropy = LabelSmoothingCrossEntropy()
def downstream_task_forward(answer_selector: EfficientModel, batch, **kwargs):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """
    try: #output result for each question
        t_batch, v_batch = batch
        vids, quess, anss, typee = v_batch
    except:
        t_batch = batch

    try: #if mixup
        x_vis_seq, frame_idxs_gt, x_txt_query, x_txt_cands, y_gt, alpha = t_batch#
    except:
        x_vis_seq, frame_idxs_gt, x_txt_query, x_txt_cands, y_gt = t_batch
        alpha = 0

    selected_frames = answer_selector(x_vis_seq, x_txt_query, x_txt_cands, **kwargs)
    y_pred = F.cosine_similarity(selected_frames.unsqueeze(1), x_txt_cands, dim=-1)  # (N, N_ans)

    loss = label_smooth_cross_entropy(y_pred/answer_selector.config.tao, y_gt, alpha)
    accs = (y_pred.argmax(dim=-1) == y_gt).float()

    try: #output result for each question,
        l_y_pred = list(y_pred.argmax(dim=-1).cpu().numpy())
        l_y_gt = list(y_gt.cpu().numpy())
        for i, (vid, ques, i_pred, i_gt, type) in enumerate(zip(vids, quess, l_y_pred, l_y_gt, typee)):
            line = f"{vid}\t{ques}\t{anss[i_pred][i]}\t{anss[i_gt][i]}\t{i_gt==i_pred}\t{type}"
            print(line)
            with open(f'{answer_selector.config.method}_{answer_selector.config.split}.csv', 'a') as f:
                f.write(line+'\n')
    except:
        pass
    return loss, accs, selected_frames, y_pred


