# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
# Modified from OmniDrive(https://github.com/NVlabs/OmniDrive)
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.models.bricks import Linear
from mmcv.models.utils import bias_init_with_prob

from mmcv.utils import force_fp32
from mmcv.core import build_assigner, build_sampler
from mmcv.core.utils.dist_utils import reduce_mean
from mmcv.core.utils.misc import multi_apply

from mmcv.models.utils import build_transformer,xavier_init
from mmcv.models import HEADS
from mmcv.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmcv.models.utils.transformer import inverse_sigmoid

from mmcv.core.bbox import build_bbox_coder
from mmcv.core.bbox.util import normalize_bbox
from mmcv.models.utils import NormedLinear

from mmcv.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from mmcv.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from mmcv.ops.iou3d_det import nms_gpu
from mmcv.core import xywhr2xyxyr
import os
from mmcv.models.utils.functional import ts2tsemb1d

from mmcv.models.bricks.transformer import build_transformer_layer_sequence,build_positional_encoding
import numpy as np
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version
import copy
from mmcv.models.utils import build_transformer,xavier_init
def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            
@HEADS.register_module()
class OrionHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 out_dims=4096,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 num_extra=256,
                 n_control=11,
                 can_bus_len=2,
                 with_mask=False,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 use_memory = False,
                 num_memory = 16,
                 scence_memory_len=256,
                 memory_decoder_transformer = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                        ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 use_col_loss = False,
                 motion_transformer_decoder=dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.0,
                        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 with_ego_pose = True, # 默认是打开egopose 和我们以前的ego pose对齐
                 bbox_coder=None,
                 test_cfg=dict(max_per_img=100),
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 class_agnostic_nms=False,
                 score_threshold=0.,
                 canbus_dropout=0.0,
                 fut_mode=6,
                 fut_ts = 6,
                 use_pe=False,
                 motion_det_score=None,
                 valid_fut_ts=6,
                 pred_traffic_light_state=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        self.with_ego_pose = with_ego_pose

        self.output_dims = out_dims
        self.n_control = n_control
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.num_extra = num_extra
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.with_mask = with_mask
        self.num_reg_fcs = num_reg_fcs
        self.test_cfg = test_cfg
        self.embed_dims = embed_dims
        self.can_bus_len = can_bus_len
        self.use_memory = use_memory
        self.num_memory = num_memory
        self.scence_memory_len = scence_memory_len
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 
        self.fut_mode = fut_mode
        self.motion_transformer_decoder = motion_transformer_decoder
        self.use_col_loss = use_col_loss
        self.use_pe = use_pe
        self.fut_ts = fut_ts
        self.motion_det_score= motion_det_score
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.traj_num_cls = 1

        super(OrionHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)
        
        self.cls_out_channels = num_classes

        self.transformer = build_transformer(transformer)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self.class_agnostic_nms = class_agnostic_nms
        self.score_threshold = score_threshold
        self.valid_fut_ts = valid_fut_ts

        if self.use_col_loss:
            # self.motion_decoder = build_transformer_layer_sequence(self.motion_transformer_decoder)
            self.motion_decoder = build_transformer(self.motion_transformer_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
                
        if self.use_memory:
            self.memory_decoder_transformer = memory_decoder_transformer
            self.memory_query = nn.Embedding(self.num_memory, self.embed_dims)
            self.scene_time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
            )
            self.memory_decoder_cq = build_transformer(self.memory_decoder_transformer)
            self.memory_decoder_mq = build_transformer(self.memory_decoder_transformer)
        self.canbus_dropout = canbus_dropout
        self.pred_traffic_light_state = pred_traffic_light_state

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        if self.use_col_loss:
            traj_branch = []
            for _ in range(self.num_reg_fcs):
                traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
                traj_branch.append(nn.ReLU())
            traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
            traj_branch = nn.Sequential(*traj_branch)
            motion_num_pred = 1
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            traj_cls_branch = []
            for _ in range(self.num_reg_fcs):
                traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
                traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
                traj_cls_branch.append(nn.ReLU(inplace=True))
            traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
            traj_cls_branch = nn.Sequential(*traj_cls_branch)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.input_projection = nn.Linear(self.in_channels, self.embed_dims)
        if self.output_dims is not None:
            self.output_projection = nn.Linear(self.embed_dims, self.output_dims)

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Embedding(self.num_extra, self.embed_dims)

        if self.output_dims is not None:
            can_bus_layers = [
                    nn.Linear(89, self.embed_dims*4), # canbus + command + egopose (b2d中的can_bus是 18维度)
                    nn.Dropout(p=self.canbus_dropout if hasattr(self,"canbus_dropout") else 0.0),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims*4, self.output_dims)]
            if not hasattr(self,"canbus_dropout") or (hasattr(self,"canbus_dropout") and self.canbus_dropout == 0):
                can_bus_layers.pop(1) # 为了与没有dropout层的早期checkpoints兼容
            self.can_bus_embed = nn.Sequential(*can_bus_layers)

        self.query_pos = None

        self.time_embedding = None

        self.ego_pose_pe = None
        
        if hasattr(self,"pred_traffic_light_state") and self.pred_traffic_light_state:
            tl_branch = []
            for _ in range(self.num_reg_fcs):
                tl_branch.append(Linear(self.embed_dims, self.embed_dims))
                tl_branch.append(nn.LayerNorm(self.embed_dims))
                tl_branch.append(nn.ReLU(inplace=True))
            tl_branch.append(Linear(self.embed_dims, 4)) # 3 + 1 , red, yellow, green, affect ego
            fc_tl = nn.Sequential(*tl_branch)
            self.tl_branches = nn.ModuleList(
                [fc_tl for _ in range(self.num_pred)])
            bias_init = bias_init_with_prob(0.01)
            for m in self.tl_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        if self.use_memory:
            # 初始化两个decoder
            for p in self.memory_decoder_cq.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.memory_decoder_mq.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        self.transformer.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
        if self.use_col_loss:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        self.sample_time = None
        self.memory_canbus = None
        self.memory_scene_tokens = None
        self.his_memory_canbus_len = None

    def pre_update_memory(self, img_metas, data):
        B = data['img_feats'].size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = data['img_feats'].new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = data['img_feats'].new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = data['img_feats'].new_zeros(B, self.memory_len, 1)
            self.memory_egopose = data['img_feats'].new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = data['img_feats'].new_zeros(B, self.memory_len, 2)
            self.sample_time = data['timestamp'].new_zeros(B)
            self.memory_canbus = data['img_feats'].new_zeros(B, self.can_bus_len, 19)
            self.his_memory_canbus_len = data['timestamp'].new_zeros(B).to(torch.int64)
            x = self.sample_time.to(data['img_feats'].dtype)
            self.memory_scene_tokens = ['' for meta in img_metas]
            if self.use_memory:
                self.memory_scene_query = data['img_feats'].new_zeros(B, self.scence_memory_len, self.embed_dims)
                self.scene_memory_timestamp = data['img_feats'].new_zeros(B, self.scence_memory_len, 1)      
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.sample_time += data['timestamp']
            x = (torch.abs(self.sample_time) < 2.0)
            y = [meta['scene_token'] == memory_tokens for meta, memory_tokens in zip(img_metas, self.memory_scene_tokens)]
            y = torch.tensor(y,device=x.device)
            x = torch.logical_and(x,y).to(data['img_feats'].dtype)
            if self.use_memory:
                self.scene_memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
                self.scene_memory_timestamp = memory_refresh(self.scene_memory_timestamp[:, :self.scence_memory_len], x)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
            for i, his_len in enumerate(self.his_memory_canbus_len):
                self.memory_canbus[i:i+1, :his_len.to(torch.int64), 1:4] -= data['can_bus'][i:i+1, :3].unsqueeze(1)
                self.memory_canbus[i:i+1, :his_len.to(torch.int64), -1] -= data['can_bus'][i:i+1, -1].unsqueeze(1)
            self.memory_canbus = memory_refresh(self.memory_canbus[:, :self.can_bus_len], x)
            self.his_memory_canbus_len = memory_refresh(self.his_memory_canbus_len, x)
            self.sample_time = data['timestamp'].new_zeros(B)
            if self.use_memory:
                self.memory_scene_query = memory_refresh(self.memory_scene_query[:, :self.scence_memory_len], x) 
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus, history_query=None):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            out_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            out_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        if self.use_memory:
            self.scene_memory_timestamp = torch.cat([torch.zeros_like(self.scene_memory_timestamp[:, :self.num_memory,:], dtype=torch.float64), self.scene_memory_timestamp], dim=1)
            self.scene_memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_canbus = torch.cat([rec_can_bus, self.memory_canbus], dim=1)
        self.his_memory_canbus_len += 1 
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        for i, his_len in enumerate(self.his_memory_canbus_len):
            self.memory_canbus[i:i+1, :his_len.to(torch.int64), 1:4] += data['can_bus'][i:i+1, :3].unsqueeze(1)
            self.memory_canbus[i:i+1, :his_len.to(torch.int64), -1] += data['can_bus'][i:i+1, -1].unsqueeze(1)
        self.memory_scene_tokens = [meta['scene_token'] for meta in img_metas]
        if self.use_memory:
            self.memory_scene_query = torch.cat([history_query.detach(), self.memory_scene_query], dim=1)
        return out_memory

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, self.n_control))) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            if self.pred_traffic_light_state:
                traffic_state_mask = [img_meta['traffic_state_mask']._data for img_meta in img_metas]
                traffic_state = [img_meta['traffic_state']._data for img_meta in img_metas]
                traffic_state_mask  = torch.cat([t for t in traffic_state_mask])
                traffic_state = torch.cat([t for t in traffic_state])
                known_traffic_state_mask = traffic_state_mask.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
                known_traffic_state = traffic_state.repeat(self.scalar, 1).long().to(reference_points.device)
                pass #TODO @DIANKUN
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query + self.num_extra
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated + self.num_extra
            tgt_size = pad_size + self.num_query + self.memory_len + self.num_extra
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            if self.with_mask:
                temporal_attn_mask[pad_size+self.num_extra:, pad_size:pad_size+self.num_extra] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            if self.pred_traffic_light_state:
                mask_dict.update(dict(known_lbs_bboxes=(known_labels, known_bboxs, known_traffic_state, known_traffic_state_mask)))
            
        else:
            attn_mask = None
            if self.with_mask:
                tgt_size = self.num_query + self.memory_len + self.num_extra
                query_size = self.num_query + self.num_propagated + self.num_extra
                attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
                attn_mask[self.num_extra:, :self.num_extra] = True
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is OrionHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(self, img_metas, pos_embed, **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # zero init the memory bank
        self.pre_update_memory(img_metas, data)
        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)

        memory = self.input_projection(memory)

        reference_points = self.reference_points.weight
        reference_points = torch.cat([torch.zeros_like(reference_points[:self.num_extra]), reference_points], dim=0)
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        query_pos = self.query_pos(nerf_positional_encoding(reference_points.repeat(1, 1, self.n_control)))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)
        
        if self.use_memory :    
            current_query = self.memory_query.weight.unsqueeze(0).repeat(B,1,1) # (4, 16, 256)
            temp_scene_query = self.memory_scene_query # (4, 256, 256)
            # time embeding
            time_embeding = self.scene_time_embedding(pos2posemb1d(self.scene_memory_timestamp).float()) # (4, 256, 1)  -> (4, 256, 256) 
            cur_time_embeding = torch.zeros_like(current_query) # (4, 16, 256) 
            # 提取历史query的信息
            # scene query <-> memory scene query
            temp_query_embedding = self.memory_decoder_mq(
                query=current_query,
                key=temp_scene_query,
                # value=temp_scene_query.permute(1, 0, 2), # (256, bs, 256)
                query_pos=cur_time_embeding,
                key_pos=time_embeding,
                # key_padding_mask=None
                )
        
        if mask_dict and mask_dict['pad_size'] > 0:
            tgt[:, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] = self.query_embedding.weight.unsqueeze(0)
            query_pos[:, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] = query_pos[:, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] * 0
        else:
            tgt[:, :self.num_extra, :] = self.query_embedding.weight.unsqueeze(0)
            query_pos[:, :self.num_extra, :] = query_pos[:, :self.num_extra, :] * 0
            

        # transformer here is a little different from PETR
        outs_dec = self.transformer(tgt, memory, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
        if mask_dict and mask_dict['pad_size'] > 0:
            reference_points = torch.cat([reference_points[:, :mask_dict['pad_size'], :], reference_points[:, mask_dict['pad_size']+self.num_extra:, :]], dim=-2)
        else:
            reference_points = reference_points[:, self.num_extra:, :]

        outs_dec = torch.nan_to_num(outs_dec)
        if mask_dict and mask_dict['pad_size'] > 0: # mask_dict['pad_size']: 100, self.num_extra: 256
            vlm_memory = outs_dec[-1, :, mask_dict['pad_size']:mask_dict['pad_size']+self.num_extra, :] # (6, 1, 1256, 256) -> (256, 256)
            outs_dec = torch.cat([outs_dec[:, :, :mask_dict['pad_size'], :], outs_dec[:, :, mask_dict['pad_size']+self.num_extra:, :]], dim=-2) # (6, 1, 1256, 256) -> # (6, 1, 1000, 256)
        else:
            vlm_memory = outs_dec[-1, :, :self.num_extra, :]
            outs_dec = outs_dec[:, :, self.num_extra:, :]
        if self.use_memory :
            # 提取当前query的信息
            # scene query <-> scene query
            history_query = self.memory_decoder_cq(
                query=temp_query_embedding,
                key=vlm_memory,
                # value=vlm_memory.permute(1, 0, 2), # (256, bs, 256)
                query_pos=None,
                key_pos=None,
                )
        
        outputs_classes = []
        outputs_coords = []
        if self.use_col_loss:
            outputs_trajs = []
            outputs_trajs_classes = []
        outputs_traffic_states = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            if self.pred_traffic_light_state:
                outputs_traffic_state = self.tl_branches[lvl](outs_dec[lvl])
                outputs_traffic_states.append(outputs_traffic_state)
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        if self.pred_traffic_light_state:
            all_traffic_states = torch.stack(outputs_traffic_states)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        rec_can_bus = data['can_bus'].clone()
        rec_can_bus[:, :3] = 0
        rec_can_bus[:, -1] = 0
        rec_can_bus = torch.cat([data['command'].unsqueeze(-1), rec_can_bus], dim=-1) # (1, 1+18)
        memory_ego_pose = self.memory_egopose.reshape(B, -1, self.topk_proposals, 4, 4).flatten(-2)
        if self.output_dims is not None:
            if self.use_memory:
                vlm_memory = torch.cat((vlm_memory, history_query), dim=1)
            vlm_memory = self.output_projection(vlm_memory) # (B, 256, 256) -> (B, 256, 4096)
            can_bus_input = torch.cat([rec_can_bus, self.memory_canbus.flatten(-2), memory_ego_pose.mean(-2).flatten(-2)], dim=-1) # (1, 19+19*2+16*2)
            can_bus_input = can_bus_input.to(torch.float32)
            can_bus_embed = self.can_bus_embed(can_bus_input) # (1, 89) -> (1, 4096)
            if self.with_ego_pose: 
                vlm_memory = torch.cat([vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2) # (1, 257, 4096)
            else:
                vlm_memory = vlm_memory # 不引入多余的can bus信息
            
        if self.use_col_loss:
            # batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # motion_query
            batch_size, num_agent = outs_dec[-1].shape[:2]
            motion_query = outs_dec[-1].permute(1, 0, 2)  # [A, B, D]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                print('not use use_pe')
                pass
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            motion_query = motion_query.permute(1, 0, 2) # batch first
            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                # value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                # key_padding_mask=invalid_motion_idx
                attn_mask=invalid_motion_idx
                )

            # ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)
            ca_motion_query = motion_hs.flatten(0, 1).unsqueeze(0)
            
            # motion_hs = motion_hs.permute(1, 0, 2).unflatten(
            #     dim=1, sizes=(num_agent, self.fut_mode)
            # )
            motion_hs = motion_hs.unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            )
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        

            outputs_traj = self.traj_branches[0](motion_hs)
            outputs_trajs.append(outputs_traj)
            outputs_traj_class = self.traj_cls_branches[0](motion_hs)
            outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
            (batch, num_agent) = motion_hs.shape[:2]
            outputs_trajs = torch.stack(outputs_trajs)
            outputs_trajs_classes = torch.stack(outputs_trajs_classes)

            
        if self.use_memory:
            # update the memory bank
            out_memory = self.post_update_memory(img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus.unsqueeze(-2), history_query)
        else:
            out_memory = self.post_update_memory(img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus.unsqueeze(-2))
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            # output_known_trajs = outputs_trajs[:, :, :mask_dict['pad_size'], :]
            # output_known_trajs_classes = outputs_trajs_classes[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            if self.use_col_loss:
                outputs_trajs = outputs_trajs[:, :, mask_dict['pad_size']:, :]
                outputs_trajs_classes = outputs_trajs_classes[:, :, mask_dict['pad_size']:, :]
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
                outs = {
                    'all_cls_scores': outputs_class,
                    'all_bbox_preds': outputs_coord,
                    'dn_mask_dict':mask_dict,
                    'all_traj_preds': outputs_trajs.repeat(outputs_coord.shape[0], 1, 1, 1, 1),
                    'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coord.shape[0], 1, 1, 1),
                }
            else:
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
                outs = {
                    'all_cls_scores': outputs_class,
                    'all_bbox_preds': outputs_coord,
                    'dn_mask_dict':mask_dict,
                }
            if self.pred_traffic_light_state:
                output_known_traffic_states = all_traffic_states[:, :, :mask_dict['pad_size'], :]
                outputs_traffic_states = all_traffic_states[:, :, mask_dict['pad_size']:, :]
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord, output_known_traffic_states)
                outs.update(dict(dn_mask_dict = mask_dict))
                outs.update(dict(all_traffic_states = outputs_traffic_states))
        else:
            if self.use_col_loss:
                outs = {
                    'all_cls_scores': all_cls_scores,
                    'all_bbox_preds': all_bbox_preds,
                    'dn_mask_dict':None,
                    'all_traj_preds': outputs_trajs.repeat(all_bbox_preds.shape[0], 1, 1, 1, 1),
                    'all_traj_cls_scores': outputs_trajs_classes.repeat(all_bbox_preds.shape[0], 1, 1, 1),
                    }
            else:   
                outs = {
                    'all_cls_scores': all_cls_scores,
                    'all_bbox_preds': all_bbox_preds,
                    'dn_mask_dict':None,
                }
            if self.pred_traffic_light_state:
                outs.update(dict(all_traffic_states = all_traffic_states))

        return outs, vlm_memory

    @force_fp32(apply_to=('preds_dicts'))
    def get_motion_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        if os.getenv('DEBUG_SHOW_PRED', None) is not None:
            score_threshold = self.score_threshold
        else:
            score_threshold = 0.
        if isinstance(score_threshold, list):
            assert len(score_threshold) == self.num_classes, \
                "score_threshold length must = class_names, len class_names: {}".format(self.num_classes)
        elif isinstance(score_threshold, dict):
            for dist_range, cls_scores_thr in score_threshold.items():
                assert len(cls_scores_thr) == self.num_classes, \
                "dist_range ---> score_threshold length must = class_names, class_names: {}".format(self.num_classes)
        else:
            score_threshold = [score_threshold] * self.num_classes

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            cur_bboxes = []
            cur_labels = []
            cur_scores = []
            cur_trajs = []
            for cid in range(self.num_classes):
                cid_mask = labels == cid
                score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
                score_mask = scores > score_thrs
                mask = cid_mask & score_mask
                cur_bboxes.append(bboxes[mask].tensor)
                cur_labels.append(labels[mask])
                cur_scores.append(scores[mask])
                cur_trajs.append(trajs[mask])

            bboxes = torch.cat(cur_bboxes)
            scores = torch.cat(cur_scores)
            labels = torch.cat(cur_labels)
            trajs =  torch.cat(cur_trajs)
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))

            if self.class_agnostic_nms and os.getenv('DEBUG_SHOW_DIR_PRED', None) is not None:
                score_weight = torch.zeros_like(labels).float()
                if isinstance(self.class_agnostic_nms, dict):
                    class_list = self.class_agnostic_nms['classes']
                    compensate = self.class_agnostic_nms['compensate']
                    class_idxes = list()
                    mask = torch.zeros_like(labels).bool()
                    for i, l in enumerate(labels):
                        for tgt_label, comp in zip(class_list, compensate):
                            if l == tgt_label:
                                score_weight[i] = comp
                                break
                    for l in class_list:
                        mask = torch.bitwise_or(mask, labels == l)
                    class_idxes = torch.where(mask)[0]
                    keep_idxes = torch.where(torch.bitwise_not(mask))[0]
                else:
                    class_idxes = torch.arange(start=0, end=labels.size()[0], step=1, device=labels.device).long()
                    keep_idxes = None
                if class_idxes.size()[0]:
                    boxes_for_nms = xywhr2xyxyr(bboxes.bev[class_idxes])
                    scores_for_nms = scores[class_idxes] + score_weight
                    # the nms in 3d detection just remove overlap boxes.
                    nms_thr = self.class_agnostic_nms['nms_thr']
                    if isinstance(nms_thr, (list, tuple)):
                        min_score, max_score = nms_thr
                        nms_thr = np.random.rand()*(max_score - min_score) + min_score
                    selected = nms_gpu(
                        boxes_for_nms.cuda(),
                        scores_for_nms.cuda(),
                        thresh=nms_thr,
                        pre_maxsize=self.class_agnostic_nms['pre_max_size'],
                        post_max_size=self.class_agnostic_nms['post_max_size'])
                    if keep_idxes is not None:
                        selected = torch.cat([class_idxes[selected], keep_idxes], 0)
                    else:
                        selected = class_idxes[selected]
                    bboxes = bboxes[selected]
                    scores = scores[selected]
                    labels = labels[selected]
                    trajs = trajs[selected]

            ret_list.append([bboxes, scores, labels, trajs])
        return ret_list

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        if os.getenv('DEBUG_SHOW_PRED', None) is not None:
            score_threshold = self.score_threshold
        else:
            score_threshold = 0.
        if isinstance(score_threshold, list):
            assert len(score_threshold) == self.num_classes, \
                "score_threshold length must = class_names, len class_names: {}".format(self.num_classes)
        elif isinstance(score_threshold, dict):
            for dist_range, cls_scores_thr in score_threshold.items():
                assert len(cls_scores_thr) == self.num_classes, \
                "dist_range ---> score_threshold length must = class_names, class_names: {}".format(self.num_classes)
        else:
            score_threshold = [score_threshold] * self.num_classes

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']

            cur_bboxes = []
            cur_labels = []
            cur_scores = []
            for cid in range(self.num_classes):
                cid_mask = labels == cid
                score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
                score_mask = scores > score_thrs
                mask = cid_mask & score_mask
                cur_bboxes.append(bboxes[mask].tensor)
                cur_labels.append(labels[mask])
                cur_scores.append(scores[mask])

            bboxes = torch.cat(cur_bboxes)
            scores = torch.cat(cur_scores)
            labels = torch.cat(cur_labels)
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))

            if self.class_agnostic_nms and os.getenv('DEBUG_SHOW_DIR_PRED', None) is not None:
                score_weight = torch.zeros_like(labels).float()
                if isinstance(self.class_agnostic_nms, dict):
                    class_list = self.class_agnostic_nms['classes']
                    compensate = self.class_agnostic_nms['compensate']
                    class_idxes = list()
                    mask = torch.zeros_like(labels).bool()
                    for i, l in enumerate(labels):
                        for tgt_label, comp in zip(class_list, compensate):
                            if l == tgt_label:
                                score_weight[i] = comp
                                break
                    for l in class_list:
                        mask = torch.bitwise_or(mask, labels == l)
                    class_idxes = torch.where(mask)[0]
                    keep_idxes = torch.where(torch.bitwise_not(mask))[0]
                else:
                    class_idxes = torch.arange(start=0, end=labels.size()[0], step=1, device=labels.device).long()
                    keep_idxes = None
                if class_idxes.size()[0]:
                    boxes_for_nms = xywhr2xyxyr(bboxes.bev[class_idxes])
                    scores_for_nms = scores[class_idxes] + score_weight
                    # the nms in 3d detection just remove overlap boxes.
                    nms_thr = self.class_agnostic_nms['nms_thr']
                    if isinstance(nms_thr, (list, tuple)):
                        min_score, max_score = nms_thr
                        nms_thr = np.random.rand()*(max_score - min_score) + min_score
                    selected = nms_gpu(
                        boxes_for_nms.cuda(),
                        scores_for_nms.cuda(),
                        thresh=nms_thr,
                        pre_maxsize=self.class_agnostic_nms['pre_max_size'],
                        post_max_size=self.class_agnostic_nms['post_max_size'])
                    if keep_idxes is not None:
                        selected = torch.cat([class_idxes[selected], keep_idxes], 0)
                    else:
                        selected = class_idxes[selected]
                    bboxes = bboxes[selected]
                    scores = scores[selected]
                    labels = labels[selected]

            ret_list.append([bboxes, scores, labels])
        return ret_list
