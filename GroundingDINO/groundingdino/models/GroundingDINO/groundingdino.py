# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import json
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from groundingdino.util import get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.text_utils import build_text_logit_mask, pad_text_logit_mask
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed
from ..registry import MODULE_BUILD_FUNCS


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
            self,
            backbone,
            transformer,
            num_queries,
            aux_loss=False,
            iter_update=True,
            query_dim=4,
            num_feature_levels=4,
            nheads=8,
            # two stage
            two_stage_type="standard",  # ['no', 'standard']
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=False,
            two_stage_bbox_embed_share=False,
            num_patterns=0,
            dn_number=0,
            dn_box_noise_scale=1.0,
            dn_label_noise_ratio=0.5,
            dn_labelbook_size=2000,
            text_encoder_type="bert-base-uncased",
            sub_sentence_present=True,
            max_text_len=256,
            anno_path=None,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        self.anno = {}
        if anno_path:
            with open(anno_path, "r") as f:
                anno = json.load(f)
                # keep unique cap
                for img, caps in anno.items():
                    for cap, items in caps.items():
                        if cap not in self.anno:
                            # delete points from items dict
                            items.pop('points', None)
                            # 'red pen': {'attribute': 'red', 'class': 'pen', 'type': 'color'}
                            self.anno[cap] = items

        super().__init__()

        # 初始化Feature Enhancer和Decoder
        self.transformer = transformer

        self.nheads = nheads
        self.num_queries = num_queries
        self.max_text_len = max_text_len
        self.hidden_dim = hidden_dim = transformer.d_model

        # Swin Transformer Backbone输出的不同尺度的特征图数量
        self.num_feature_levels = num_feature_levels

        # GroundingDINO的子句级文本token attention机制
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        # 确认输出为四维框 (cx, cy, w, h)
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        # Denoising Training参数，详见DINO模型
        # 本项目中dn_number=0，该功能未启用
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        # pooler是当前模型不使用的部分，因此默认不训练它
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        # 映射bert token维度到模型的hidden_dim维度
        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.feat_map.weight, gain=1)
        nn.init.constant_(self.feat_map.bias, 0)

        # special tokens
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        # We extract three image feature scales, from 8× to 32×.
        # It is named “4scale” in DINO since we downsample the 32× feature map to 64× as an extra feature scale.
        # Swin-T Transformer Backbone 返回8×、16×、32×三个尺度的特征图
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            in_channels = 0
            # 先建立三个不同输入通道数的1x1卷积层，将8×、16×、32×三个尺度的特征图映射到hidden_dim维度
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            # 再用一个kernel_size=3, stride=2, padding=1的卷积层将32×的特征图的空间分辨率再缩小一半，等效为一个64x的特征图
            # 同时映射到hidden_dim维度
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # 虽然num_feature_levels=1在数学上不是不可行，但是 two-stage proposal 要求必须使用多尺度特征图
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Swin Transformer Backbone
        self.backbone = backbone

        # 对 decoder 的中间层预测也计算检测损失，而不只监督最后一层
        # 即使设置aux_loss=True，相关代码在forward中也未执行推理
        # 当前实现等效于强制为False
        self.aux_loss = aux_loss
        self.box_pred_damping = None

        # iterative reference point update
        # Decoder 的每一层是否根据当前层的输出，预测边界框增量，并更新下一层使用的 reference box
        # 该参数为历史遗留，实际上未使用
        # 当前代码实现由 decoder 中的 bbox_embed 和 reference 更新逻辑实际完成，强制启用
        # 这里的 assert 主要用于强制保证调用者没有传入当前实现不支持的 False
        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share

        # prepare class & box embed

        # _class_embed 中没有可学习参数，其实际为 视觉-语言相似度计算模块
        _class_embed = ContrastiveEmbed(self.max_text_len)
        # _bbox_embed 是3层MLP，激活函数为relu
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 对bbox head除最后一层之外的 Linear 层进行 Xavier uniform 初始化
        for layer in _bbox_embed.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)
        # bbox head的最后一层线性层 Linear(hidden_dim, 4) 使用全0初始化
        # 这是一种 identity initialization（恒等映射初始化）
        # 开始时：新框 = 旧框 + 0
        # 训练后：新框 = 旧框 + 学习到的修正量
        # 提高训练稳定性
        nn.init.constant_(_bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias, 0)

        # 所有 decoder layer 的 bbox head 是否使用同一个 MLP
        if dec_pred_bbox_embed_share:
            # 共享参数（当前配置）
            box_embed_layerlist = [_bbox_embed for _ in range(transformer.num_decoder_layers)]
        else:
            # 不共享参数
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for _ in range(transformer.num_decoder_layers)
            ]

        # 所有 decoder layer 的 class head 始终共享参数，但其中实际上没有可学习参数
        class_embed_layerlist = [_class_embed for _ in range(transformer.num_decoder_layers)]

        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two-stage proposal
        # two_stage_type="no"：query 来自可学习 embedding
        # two_stage_type="standard"：query 来自 encoder 生成的 top-k proposal
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )

        # 如果启动two-stage proposal
        if two_stage_type != "no":
            # encoder proposal bbox head 是否与 decoder 的 bbox head 共享参数
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                # 当前配置是不共享参数
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            # encoder proposal class head 是否与 decoder 的 class head 共享参数
            # 由于 _class_embed 中没有可学习参数，因此 two_stage_class_embed_share 实际上对模型没有影响
            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

    def set_image_tensor(self, samples: NestedTensor | List[torch.Tensor]):
        # 缓存features和poss
        # 使得图像backbone forward一次，即可用于多个caption的forward
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 执行 backbone forward
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self, 'poss'):
            del self.poss

    def set_image_features(self, features, poss):
        self.features = features
        self.poss = poss

    def forward(self, samples: List[torch.Tensor], captions: List[str], **kw):
        """
        该函数输出参数已改进，现在可以输入由不同shape的image tensor组成的list

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        # 类型转换提前
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # 兼容之前的调用参数
        if kw.get('targets', None) is not None:
            captions = [t["caption"] for t in kw["targets"]]

        # split captions to subject and context
        # subject 与低层 image tokens 做 cross-attention
        # subject 是物体的基本特征属性，比如颜色、年龄、性别、动作等
        # context 与高层 image tokens 做 cross-attention
        # 只有 location 会被视为是 context，并在split_caption函数中单独处理
        # 因为作者认为 location 是一种关系属性，涉及到与场景中其他区域或物体之间的联系，而不是可以仅从目标物体局部外观判断的普通属性
        # attribute 用于 forward 之后的 contrastive loss，同类别但不同 attribute 的文本 embedding 作为负样本
        # split_caption 返回的字符串末尾加.是为了沿用GroundingDINO的格式，后续会被去掉
        subjects, contexts, attributes = [], [], []
        for caption in captions:
            subject, context, att = split_caption(caption, self.anno)
            subjects.append(subject)  # ['blue box.', 'yellow box.', '...', '...']
            contexts.append(context)  # ['on table.', 'on ground.', '...', '...']
            attributes.append(att)  # ['blue.','yellow.', '...', '...']

        # encoder texts
        # Keep offsets only for constructing exact role masks
        # they are removed before calling BERT
        tokenized = self.tokenizer(
            captions,
            padding="longest",
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        ).to(samples.device)

        # 实现子句级文本token attention机制
        # text_self_attention_masks，[bs, seq_len, seq_len]
        # mask[b, i, j] = True：第 i 个 token 可以注意第 j 个 token
        # position_ids，[bs, seq_len]
        # 每个子句内部的位置编号重新从 0 开始
        # cate_to_token_mask_list，长度为 bs 的 list，每个元素为 [num_clauses_b, seq_len]
        # [num_clauses_b, seq_len] 中每一行对应一个子句，用 True 标记该子句中的普通文本 token
        # special_tokens和句号、问号等分隔 token 则被标记为False
        # cate_to_token_mask_list 已废弃，通过其他方式实现
        (text_self_attention_masks, position_ids, cate_to_token_mask_list) = (
            generate_masks_with_special_tokens_and_transfer_map(tokenized, self.special_tokens)
        )

        # text 过长则执行截断
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]
            tokenized["special_tokens_mask"] = tokenized["special_tokens_mask"][:, : self.max_text_len]
            tokenized["offset_mapping"] = tokenized["offset_mapping"][:, : self.max_text_len]

        # extract text embeddings
        # offsets and special-token metadata are not valid BERT inputs
        # they will be removed before the encoder call
        if self.sub_sentence_present:
            # 开启子句级文本token attention机制时
            # 默认的attention_mask需要替换为text_self_attention_masks
            # 还需要加入position_ids
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items()
                if k not in {"attention_mask", "special_tokens_mask", "offset_mapping"}
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items()
                if k not in {"special_tokens_mask", "offset_mapping"}
            }

        bert_output = self.bert(**tokenized_for_encoder)  # bs, seq_len, 768
        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, seq_len, d_model

        # text_token_mask: True for nomask, False for mask
        text_token_mask = tokenized["attention_mask"].bool()  # bs, seq_len
        text_logit_mask = pad_text_logit_mask(
            build_text_logit_mask(self.tokenizer, tokenized), self.max_text_len
        )
        special_token_mask = tokenized["special_tokens_mask"].bool()  # bs, seq_len

        # 计算 subject、context、attribute 分别对应原 caption 中哪些 token
        text_subject_mask = _role_mask(
            captions, subjects, tokenized["offset_mapping"],
            text_token_mask, special_token_mask
        )
        text_context_mask = _role_mask(
            captions, contexts, tokenized["offset_mapping"],
            text_token_mask, special_token_mask
        )
        text_attribute_mask = _role_mask(
            captions, attributes, tokenized["offset_mapping"],
            text_token_mask, special_token_mask
        )

        # The input sequence was already truncated before BERT
        # assert encoded_text.shape[1] <= self.max_text_len 必然为True
        # Keep this guard only for encoder implementations that may alter sequence length
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            text_subject_mask = text_subject_mask[:, : self.max_text_len]
            text_context_mask = text_context_mask[:, : self.max_text_len]
            text_attribute_mask = text_attribute_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, seq_len, d_model
            "text_token_mask": text_token_mask,  # bs, seq_len
            "text_logit_mask": text_logit_mask,  # bs, max_text_len
            "position_ids": position_ids,  # bs, seq_len
            "text_self_attention_masks": text_self_attention_masks,  # bs, seq_len, seq_len
            "text_subject_mask": text_subject_mask,  # bs, seq_len
            "text_context_mask": text_context_mask,  # bs, seq_len
            "text_attribute_mask": text_attribute_mask,  # bs, seq_len
        }

        # import ipdb; ipdb.set_trace()
        # 执行 backbone forward，并缓存生成的不同尺度的特征图
        if not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)

        # encoder images
        srcs = []  # sources
        masks = []
        for l, feat in enumerate(self.features):
            # 不同尺度的特征图通道数也不同，通过1x1卷积映射到同一维度
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            assert mask is not None
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            # 当 Backbone 输出的特征层数少于模型要求的特征层数时
            # 对最后一层特征进行下采样，补充更低分辨率、更大感受野的特征层
            _len_srcs = len(srcs)
            # 从已有特征层的下一个编号开始，一直生成到目标层数
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    # 第一层额外特征的输入应当是 Backbone 的最后一个原始输出
                    src = self.input_proj[l](self.features[-1].tensors)
                else:
                    # 生成过至少一层额外特征，则把前一次生成的特征作为输入
                    src = self.input_proj[l](srcs[-1])
                # 取出原始输入图像的 padding mask
                m = samples.mask
                # 通过二维最近邻插值，将原始图像分辨率下的 mask 缩放到新增特征图的空间尺寸
                mask = (F.interpolate(m.unsqueeze(0).float(), size=src.shape[-2:], mode="nearest")
                        .squeeze(0).to(torch.bool))
                # 调用backbone的位置编码模块，为新增的特征图生成位置编码
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                # 将新增的特征图添加到已有特征列表中
                srcs.append(src)
                masks.append(mask)
                self.poss.append(pos_l)

        # input_query_bbox，外部提供的 query reference box
        # input_query_label，外部提供的 query content 特征
        # attn_mask，query 之间的 self-attention mask，
        # 用于 dn training 时防止 denoising query 与普通 query 互相看到
        # 当前 dn training 未启用
        input_query_bbox = input_query_label = attn_mask = None

        # hs：decoder 每层输出的 content query，[num_decoder_layers, bs, num_queries, 256]
        # reference：各阶段的 reference boxes，[num_decoder_layers + 1, bs, num_queries, 4]
        # hs_enc：Encoder 选出的 top-k proposal 对应的视觉特征，[1, bs, num_queries, 256]
        # ref_enc：Encoder 选出的 top-k proposal box，[1, bs, num_queries, 4]
        # init_box_proposal：Encoder 选出的原始 top-k proposal box，未经过 encoder bbox head 的修正，[bs, num_queries, 4]
        # img_embs：经过 Encoder 图文交互后的 image 特征，经过选择与融合后作为 query，[bs, num_queries, 256]
        # txt_embs：经过 Encoder 图文交互后的文本特征，[bs, seq_len, 256]
        hs, reference, hs_enc, ref_enc, init_box_proposal, img_embs, txt_embs = self.transformer(
            srcs, masks, input_query_bbox, self.poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        # 取每个 decoder layer 输入的 reference box（排除 decoder 最后一层输出的 reference box）
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
        ):
            # 计算每层的 bbox 的预测偏移量
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            # 与原始 reference box 相加，得到最终的预测 bbox
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs = layer_outputs_unsig.sigmoid()
            # 逐层保存预测 bbox
            outputs_coord_list.append(layer_outputs)
        # outputs_coord_list：[num_decoder_layers, bs, num_queries, 4]
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        # class_embed 的输出是每一个 query 与每一个text token 的匹配分数，[pad] 对应-inf
        # 最后用-inf填充到最大长度；outputs_class：[num_decoder_layers, bs, num_queries, seq_len]
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        # 基本输出：最后一层 query-token 匹配分数和预测 bbox
        # out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # # for intermediate outputs
        # 整理前num_decoder_layers-1层的输出，用于aux loss的计算
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        token_masks = text_dict["text_attribute_mask"]
        out = {
            "pred_logits": outputs_class[-1],  # 最后一层 decoder 输出的 query-token 匹配分数
            "pred_boxes": outputs_coord_list[-1],  # 最后一层 decoder 输出的预测 bbox
            "img_embs": hs[-1],  # 最后一层 decoder 输出的 query 特征
            "txt_embs": txt_embs,  # 经过 encoder 图文交互后的文本特征
            "token_masks": token_masks,  # 也是 attribute_token_mask，为了兼容
            "text_token_mask": text_token_mask,  # 有效文本 token mask
            "text_logit_mask": text_logit_mask,  # 分类头使用的 token mask
            "attribute_token_mask": text_dict["text_attribute_mask"]  # attribute token mask
        }

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     # encoder 选出的 top-k proposal 对应的 content 特征
        #     interm_coord = ref_enc[-1]
        #     # encoder 选出的 top-k proposal 对应的 query-token 匹配分数
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor()  # If necessary
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


def split_caption(caption, anno=None):
    if anno is not None:
        found = [(cap, items['type'], items['class'], items['attribute']) for cap, items in anno.items() if
                 cap.lower() + '.' == caption.lower()]
        if found:
            cap, typ, cls, att = found[0]
            if typ == 'location':
                return cls + '.', att + '.', att + '.'
            else:
                return caption, "", att + '.'
        else:
            print("CAPTION NOT FOUND IN ANNO")
            return caption, "", ""

    else:
        return caption, "", ""


def _role_mask(captions, role_texts, offsets, attention, special):
    """将每条 caption 中指定语义角色对应的字符区间映射为 token mask。

    ``role_texts`` 通常是 ``split_caption`` 提取出的 subject、context 或 attribute 文本。
    由于这些文本仍是字符串，而 Transformer 后续只能按 token 选择文本特征，
    因此需要借助 tokenizer 返回的 ``offset_mapping``，将角色文本在原 caption 中的字符范围转换成 token 位置。

    Args:
        captions: 长度为 bs 的原始 caption 列表。
        role_texts: 长度为 bs 的角色文本列表；每项对应同一行 caption 中的
            subject、context 或 attribute。空字符串表示该样本没有此角色。
        offsets: 形状为 [bs, seq_len, 2] 的 offset mapping。``offsets[b, i]``
            给出第 i 个 token 在原始 caption 中对应的半开字符区间 ``[token_start, token_end)``。
        attention: 形状为 [bs, seq_len] 的 tokenizer attention mask。
            True 表示有效 token，False 表示 padding。
        special: 形状为 [bs, seq_len] 的 special-token mask。
            True 表示 [CLS]、[SEP]、[PAD] 等特殊 token。

    Returns:
        形状为 [bs, seq_len] 的布尔 mask。
        True 表示该 token：
        1. 与角色文本在 caption 中的字符区间有重叠；
        2. 是有效的非 padding token；
        3. 不是特殊 token。

        WordPiece 拆分出的多个子 token 只要与角色字符区间相交，都会被标记为 True。
        若角色文本为空或未在 caption 中找到，则该行全为False；
        若相同文本出现多次，当前实现只匹配第一次出现的位置。
    """
    # 先建立与 tokenizer attention mask 同形状的全 False 占位
    masks = torch.zeros_like(attention, dtype=torch.bool)
    for row, (caption, role_text) in enumerate(zip(captions, role_texts)):
        # 该样本没有可对应的 role_texts 时保持整行全 False
        if not role_text:
            continue

        # split_caption 返回的文本通常以句号结尾。去掉末尾句号后再查找
        # 避免把分隔符本身误标为 subject/context/attribute token
        caption_body = caption.rstrip().rstrip(".")
        role_body = role_text.strip().rstrip(".")

        # 使用不区分大小写的字符串查找，得到角色在 caption 中第一次出现的字符区间 [start, end)。
        # 这里只改变大小写，不改变字符串长度，因而索引仍与 tokenizer 针对原 caption 生成的 offsets 对齐。
        start = caption_body.lower().find(role_body.lower())
        if start >= 0 and role_body:
            end = start + len(role_body)
            row_offsets = offsets[row]
            for token_index, (token_start, token_end) in enumerate(row_offsets.tolist()):
                # 当且仅当 token_start < end 且 token_end > start 时
                # 两个半开区间 [token_start, token_end) 与 [start, end)相交
                # token_end == token_start 的特殊 token 不参与匹配
                if token_end > token_start and token_start < end and token_end > start:
                    masks[row, token_index] = True

        # 最后再与有效 token mask 取交集，并排除 [CLS]/[SEP]/[PAD] 等特殊 token
        # 保证输出可以安全地用于选择真实文本 embedding
        masks[row] &= attention[row] & ~special[row]
    return masks


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=False,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        anno_path=getattr(args, "anno_path", None),
    )

    return model
