# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import warnings
from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torch.nn import functional as F

from groundingdino.util.misc import inverse_sigmoid
from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)


class Transformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_queries=900,
            num_encoder_layers=6,
            num_unicoder_layers=0,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.0,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
            query_dim=4,
            num_patterns=0,  # 为每个 query 增加多个 pattern embedding
            # for deformable encoder
            num_feature_levels=4,
            enc_n_points=4,
            dec_n_points=4,
            # init query
            learnable_tgt_init=True,
            # two stage
            two_stage_type="standard",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
            embed_init_tgt=True,
            # for text
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_checkpoint=True,
            use_transformer_ckpt=True,
            use_text_cross_attention=True,
            text_dropout=0.0,
            fusion_dropout=0.0,
            fusion_droppath=0.1,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        # 构造 image encoder layer
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points,
        )

        if use_text_enhancer:
            # text 特征增强层
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            # image 与 text 双向cross attention
            # 图像 token → 读取文本信息 → 更新图像 token
            # 文本 token → 读取图像信息 → 更新文本 token
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # 当前Encoder中硬编码实现了 post norm，即先残差相加，再 LayerNorm
        assert encoder_norm is None

        # 将前面创建的三类 layer 交给 TransformerEncoder
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            enc_layer_share=False,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        # 构造 image decoder layer
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
            use_text_feat_guide=False,
            # Query-to-text cross-attention
            use_text_cross_attention=use_text_cross_attention
        )

        decoder_norm = nn.LayerNorm(d_model)
        # 创建 decoder
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            warnings.warn("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if num_patterns > 0:
            self.patterns = nn.Embedding(num_patterns, d_model)
        else:
            self.patterns = None

        # 为每个 feature level 创建一个可学习向量
        # 图像 token 的位置编码实际是：二维空间位置编码 + 特征层级编码
        # 使得 Transformer 不仅知道 token 位于哪个空间位置，还知道它来自哪个尺度
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        # 强制要求使用可学习参数初始化 content query
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"

        # 使用可学习参数初始化 content query
        self.embed_init_tgt = embed_init_tgt
        # 启用两阶段推理且使用可学习参数初始化 content query      或者      启用两阶段推理
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            # 创建 content query
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            # 用于进一步映射和归一化 Encoder 输出的image token
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            # 不使用可学习的宽高 embedding
            # proposal 的宽高由 gen_encoder_output_proposals() 中的固定规则生成
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            # 此时为每个 query 创建一个可学习的四维 reference box 用于初始化
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        # 暂时设置为 None，稍后由 GroundingDINO.__init__() 赋值
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # 用于 lower image tokens 和 higher image tokens 的 cross attention
        self.cross_attention = CrossAttentionLayer(d_model)

        # 归一化 Encoder top-k proposal 特征
        self.tgt_fusion_norm = nn.LayerNorm(d_model)
        # 然后与 learned query 进行加权求和，权重为可学习参数
        # 初始值较低，减少对预训练基模的影响
        self.tgt_fusion_alpha = nn.Parameter(torch.tensor(0.1))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                # 其中 sampling offset 的 weight 初始化为 0
                # 但 bias 被初始化成不同方向的采样网格
                # 并且不同采样点的距离逐渐增大
                # 使得模型在训练初期已经具有合理的多尺度局部采样模式
                m._reset_parameters()
        # 查表（lookup table）型可学习向量使用正态分布初始化
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)
        if self.tgt_embed is not None:
            nn.init.normal_(self.tgt_embed.weight)
        if self.patterns is not None:
            nn.init.normal_(self.patterns.weight)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None):
        """
        Input:
            - srcs: 多尺度图像特征 List，每个元素形状为 [bs, ci, hi, wi]，list 长度为 num_feature_levels
            - masks: 多尺度图像特征对应的 padding mask，每个元素形状为 [bs, hi, wi]，list 长度为 num_feature_levels
            - refpoint_embed: 外部输入的 query 对应的 reference box，用于 dn training [bs, num_dn, 4]
            - pos_embeds: 多尺度图像特征对应的 position embedding，每个元素形状为 [bs, ci, hi, wi]，list 长度为 num_feature_levels
            - tgt: 外部输入的 query content embedding，用于 dn training [bs, num_dn, d_model]
            - attn_mask: Decoder 中 query self-attention 使用的 mask，用于 dn training [bs, num_dn, num_dn]
            - text_dict: 文本信息字典

        Output:
            - hs：decoder 每层输出的 content query，[num_decoder_layers, bs, num_queries, c]
            - reference：各阶段的 reference boxes，[num_decoder_layers + 1, bs, num_queries, 4]
            - hs_enc：Encoder 选出的 top-k proposal 对应的视觉特征，[1, bs, num_queries, c]
            - ref_enc：Encoder 选出的 top-k proposal box，[1, bs, num_queries, 4]
            - init_box_proposal：Encoder 选出的原始 top-k proposal box，未经过 encoder bbox head 的修正，[bs, num_queries, 4]
            - img_embs：经过 Encoder 图文交互后的 image 特征，经过选择与融合后作为 query，[bs, num_queries, c]
            - txt_embs：经过 Encoder 图文交互后的文本特征，[bs, seq_len, c]


        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # 对于每层特征图
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # 记录尺寸
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # 展平为 image token 序列
            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            # 叠加位置编码
            # pos_embed：描述 token 位于该特征图中的二维位置
            # level_embed：描述 token 来自哪一个特征尺度
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            # 加入列表
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # 拼接为完整的 image token 序列
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hw}, c
        # 每层特征图的尺寸 [num_feature_levels, 2]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        # 每层特征图的 token 在完整序列中的起始索引 [num_feature_levels]
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # 每张图片的各尺度特征图的有效宽和高占比 [bs, num_feature_levels, 2]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src=src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            # we ~ the mask . False means use the token; True means pad the token
            # 为符合 PyTorch attention 的 key_padding_mask 语义，mask 需要取反
            text_attention_mask=~text_dict["text_token_mask"],
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        #########################################################
        # End Encoder
        # - memory: [bs, \sum{hw}, c]，特征增强后的 image token
        # - memory_text: [bs, seq_len, c]，特征增强后的 text token
        #########################################################

        # 保存更新后的 text 特征
        text_dict["encoded_text"] = memory_text
        txt_embs = text_dict["encoded_text"]

        if self.two_stage_type == "standard":
            # 为每个 Encoder 输出的 image token 生成候选框
            # 以每个 image token 在对应特征图中的位置为中心，不同的特征图尺度生成不同大小的候选框
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            # 进一步映射和归一化 Encoder 输出的 image token
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            # 计算所有 image token 与每个 text token 的相似度
            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            # 取出 [CLS] token 的 匹配分数作为排序分数
            topk_logits = enc_outputs_class_unselected[:, :, 0]  # (bs, \sum{hw})

            # 修正后的边界框 = bbox head 预测增量 + 初始边界框
            enc_outputs_coord_unselected = (
                    self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid

            # 选中的 image token 的索引 [bs, num_queries]
            topk = self.num_queries
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, num_queries

            """
            # 按照索引大小进行排序
            # 由于图像 token 是按照 feature level 顺序拼接的，因此较大的索引通常更可能来自后面的大视野特征层
            # 前 10% 为 higher tokens, 后 90% 为 lower tokens
            lower_idxes, higher_idxes = split_tokens(topk_proposals, 0.1)
            # 取出对应的 token
            lower_tokens = torch.gather(output_memory, 1, lower_idxes.unsqueeze(-1).expand(-1, -1, self.d_model))
            higher_tokens = torch.gather(output_memory, 1, higher_idxes.unsqueeze(-1).expand(-1, -1, self.d_model))

            #   lower_tokens 以 subject 文本为 Key/Value 做 cross-attention
            #   higher_tokens 以 context 文本为 Key/Value 做 cross-attention
            lower_tokens, higher_tokens = _apply_subject_and_context_attention(
                self.cross_attention,
                lower_tokens,
                higher_tokens,
                text_dict["encoded_text"],
                text_dict["text_subject_mask"],
                text_dict["text_context_mask"],
            )
            # lower token 会进一步吸收 higher token 中的上下文信息
            updated_lower_tokens = self.cross_attention(lower_tokens, higher_tokens)
            # 将更新后的 lower token 按原位置写回，higher token 不做修改
            output_memory = output_memory.scatter(1, lower_idxes.unsqueeze(-1).expand(-1, -1, self.d_model),
                                                  updated_lower_tokens)
            """

            # gather boxes
            # 取出 Encoder 输出的 proposal
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoid
            # 不使其梯度回传
            refpoint_embed_ = refpoint_embed_undetach.detach()
            # 取出 Encoder bbox head 修正之前的原始 proposal
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            # 取出选中的 Encoder 输出的 image token，作为 content query，[bs, num_queries, c]
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            if self.embed_init_tgt:
                # 取出可学习 query
                learned_tgt = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )
                # 归一化，加权融合，权重为可学习参数
                fused_tgt = self.tgt_fusion_norm(tgt_undetach)
                tgt_ = learned_tgt + self.tgt_fusion_alpha * fused_tgt
            else:
                tgt_ = tgt_undetach

            if refpoint_embed is not None:
                # 拼接外部输入的 refpoint_embed 和 tgt；用于 dn training
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                # 当前配置，不使用外部输入的 refpoint_embed 和 tgt
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            # 使用可学习的 query 和 候选框
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                # 同上
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_
            init_box_proposal = refpoint_embed_.sigmoid()

            if self.num_patterns > 0:
                # 将每个 query 扩展为 num_patterns 份
                # 实现对同一个位置的多种 content 的检测
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))

        # 保存更新后的 image 特征
        img_embs = tgt
        #########################################################
        # End preparing tgt
        # - tgt: bs, num_queries, d_model
        # - refpoint_embed(unsigmoid): bs, num_queries, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),  # [length, batch, channel]
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),  # [length, batch, channel]
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            # we ~ the mask . False means use the token; True means pad the token
            # 为符合 PyTorch attention 的 key_padding_mask 语义，mask 需要取反
            text_attention_mask=~text_dict["text_token_mask"],
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, num_queries, d_model 每层输出的 content query
        # references: n_dec+1, bs, num_queries, query_dim 各阶段的 reference boxes
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        # 整理 Encoder 输出的中间结果，准备返回
        if self.two_stage_type == "standard":
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (1, bs, num_queries, d_model)
        # ref_enc: (1, bs, num_queries, query_dim)
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal, img_embs, txt_embs
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (1, bs, nq, d_model)
        # ref_enc: sigmoid coordinates. (1, bs, nq, query_dim)
        # init_box_proposal：Encoder 选出的原始 top-k proposal box，未经过 encoder bbox head 的修正，[bs, num_queries, 4]
        # img_embs：经过 Encoder 图文交互后的 image 特征，经过选择与融合后作为 query，[bs, num_queries, 256]
        # txt_embs：经过 Encoder 图文交互后的文本特征，[bs, seq_len, 256]


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_layer,
            num_layers,
            d_model=256,
            num_queries=900,
            enc_layer_share=False,
            text_enhance_layer=None,
            feature_fusion_layer=None,
            use_checkpoint=True,
            use_transformer_ckpt=True,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 900.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []

        # 如果 num_layers > 0，则创建对应数量的 encoder_layer 和 text_enhance_layer 和 feature_fusion_layer
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer
            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # deformable attention 的参考点生成器
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 为每个 feature level 建立规则网格
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij"
            )
            # 展平空间网格，然后基于真实图像区域把坐标归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 拼接多个 feature level
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 扩展到所有 feature level
        return reference_points

    def forward(
            self,
            # for images
            src: Tensor,
            pos: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
            key_padding_mask: Tensor,
            # for texts
            memory_text: Tensor = None,
            text_attention_mask: Tensor = None,
            pos_text: Tensor = None,
            text_self_attention_masks: Tensor = None,
            position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=self.d_model, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                        use_reentrant=False
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                    use_reentrant=False
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer,
            num_layers,
            norm=None,
            return_intermediate=True,
            d_model=256,
            query_dim=4,
            num_feature_levels=4,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim == 4, "query_dim should be 4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None
        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.d_model = d_model
        self.ref_anchor_head = None

    def forward(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
            # for memory
            level_start_index: Optional[Tensor] = None,  # num_levels
            spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
            valid_ratios: Optional[Tensor] = None,
            # for text
            memory_text: Optional[Tensor] = None,
            text_attention_mask: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :],
                num_pos_feats=self.d_model // 2,
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
            self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
            use_text_feat_guide=False,
            use_text_cross_attention=True,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.amp.autocast('cuda', enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
            self,
            # for tgt
            tgt: Optional[Tensor],  # nq, bs, d_model
            tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
            tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
            memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
            text_attention_mask: Optional[Tensor] = None,  # bs, num_token
            # for memory
            memory: Optional[Tensor] = None,  # hw, bs, d_model
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_level_start_index: Optional[Tensor] = None,  # num_levels
            memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
            memory_pos: Optional[Tensor] = None,  # pos for memory
            # sa
            self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
            cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, lower_tokens, higher_tokens, lower_mask=None, V_mask=None):
        Q = lower_tokens.transpose(0, 1)
        K = higher_tokens.transpose(0, 1)
        V = higher_tokens.transpose(0, 1)
        attn_output, _ = self.cross_attention(Q, K, V, key_padding_mask=V_mask)
        # Add & norm
        updated_lower_tokens = self.norm(attn_output.transpose(0, 1) + lower_tokens)
        return updated_lower_tokens


def _pad_selected_text_tokens(encoded_text, selection_mask):
    # 选择每个样本的文本 token
    selected_text_tokens = [text[mask] for text, mask in zip(encoded_text, selection_mask)]
    # 补齐不同长度的 token 序列
    max_size = encoded_text.size(1)
    padded_text_tokens = torch.stack(
        [F.pad(tokens, (0, 0, 0, max_size - tokens.size(0))) for tokens in selected_text_tokens]
    )
    # 记录每一行有多少个有效 token
    valid_lengths = selection_mask.sum(dim=1)
    positions = torch.arange(max_size, device=encoded_text.device).unsqueeze(0)
    # 构造 key_padding_mask
    key_padding_mask = positions >= valid_lengths.unsqueeze(1)
    return padded_text_tokens, key_padding_mask


def _apply_text_cross_attention(
        cross_attention,
        query_tokens,
        encoded_text,
        selection_mask,
        active_rows=None,
):
    if active_rows is None:
        active_rows = torch.ones(query_tokens.size(0), dtype=torch.bool, device=query_tokens.device)
    # 找到有效的行
    active_indices = torch.nonzero(active_rows, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return query_tokens
    # 只取 active 行的文本和 mask
    active_text, key_padding_mask = _pad_selected_text_tokens(
        encoded_text.index_select(0, active_indices),
        selection_mask.index_select(0, active_indices),
    )
    updated_tokens = cross_attention(
        # 只取 active 行的 query
        query_tokens.index_select(0, active_indices),
        active_text,
        V_mask=key_padding_mask,
    )
    # 把 active 行的结果写回原 batch
    return query_tokens.index_copy(0, active_indices, updated_tokens)


def _apply_subject_and_context_attention(
        cross_attention,
        lower_tokens,
        higher_tokens,
        encoded_text,
        subject_mask,
        context_mask,
):
    lower_tokens = _apply_text_cross_attention(
        cross_attention,
        lower_tokens,
        encoded_text,
        subject_mask,
    )
    # 跳过没有 context token 的样本，避免把所有 Key 都屏蔽
    valid_context_rows = context_mask.any(dim=1)
    higher_tokens = _apply_text_cross_attention(
        cross_attention,
        higher_tokens,
        encoded_text,
        context_mask,
        valid_context_rows,
    )
    return lower_tokens, higher_tokens


def split_tokens(topk_proposals, radio_higher=0.1):
    sorted = torch.sort(topk_proposals, dim=1, descending=False)[0]
    num_lower = int((1 - radio_higher) * topk_proposals.size(1))
    lower_idxes = sorted[:, :num_lower]
    higher_idxes = sorted[:, num_lower:]

    return lower_idxes, higher_idxes
