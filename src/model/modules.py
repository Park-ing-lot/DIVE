import copy
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bart import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    invert_mask,
    EncoderLayer,
    LayerNorm,
)

from src.model.config import MultiModalBartConfig


class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim):
        super(ImageEmbedding, self).__init__()
        self.linear = nn.Linear(image_dim, final_dim)
        self.layernorm = nn.LayerNorm(final_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, image_features):
        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            embedded = self.linear(img_tensor)
            embedded = self.layernorm(embedded)
            embedded = self.dropout(embedded)
            
        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index: index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output


# This is copied from transformers.BartEncoder
# The modifications are:
# - added embed_images layer
# - added _embed_multi_modal function
# - added image_features in forward
class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """

    def __init__(self, config: MultiModalBartConfig, embed_tokens):
        super().__init__()

        self.img_feat_id = config.img_feat_id # 50273
        self.cls_token_id = config.cls_token_id
        self.end_img_id = 50266

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(config.image_feature_size, embed_dim)
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

        self.person_ids = [i + 50295 for i in range(50)]

    def _embed_multi_modal(self, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        tmp = (input_ids == self.end_img_id).nonzero(as_tuple=True)
        
        mask = (input_ids == self.img_feat_id) | (input_ids == self.cls_token_id) | \
            (input_ids == 50295) | (input_ids == 50296) | (input_ids == 50297) | (input_ids == 50298) | (input_ids == 50299) | (input_ids == 50300) |\
            (input_ids == 50301) | (input_ids == 50302) | (input_ids == 50303) | (input_ids == 50304) | (input_ids == 50305) | (input_ids == 50306) |\
            (input_ids == 50307) | (input_ids == 50308) | (input_ids == 50309) | (input_ids == 50310) | (input_ids == 50311) | (input_ids == 50312) |\
            (input_ids == 50313) | (input_ids == 50314) | (input_ids == 50315) | (input_ids == 50316) | (input_ids == 50317) | (input_ids == 50318)
            # (input_ids == 50319) | (input_ids == 50320) | (input_ids == 50321) | (input_ids == 50322) | (input_ids == 50323) | (input_ids == 50324) |\
            # (input_ids == 50325) | (input_ids == 50326) | (input_ids == 50327) | (input_ids == 50328) | (input_ids == 50328) | (input_ids == 50329) |\
            # (input_ids == 50330) | (input_ids == 50331) | (input_ids == 50332) | (input_ids == 50333) | (input_ids == 50334) | (input_ids == 50335) |\
            # (input_ids == 50336) | (input_ids == 50337) | (input_ids == 50338) | (input_ids == 50339) | (input_ids == 50340) | (input_ids == 50341) |\
            # (input_ids == 50342) | (input_ids == 50343) | (input_ids == 50344) | (input_ids == 50345)
        # raise
        for i, ma in enumerate(mask):
            mask[i][tmp[1][i]:] = False
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)

        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] += value

        return embedded

    def forward(
            self,
            input_ids,
            image_features,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False
    ):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """

        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        # encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions
