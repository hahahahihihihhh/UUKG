import json
import pickle

from torch import nn
import torch
import numpy as np

device = torch.device('cuda')

config_file = 'config.json'
with open(config_file) as config:
    config = json.load(config)

ke_model, ke_dim  = config["ke_model"], config["ke_dim"]
max_hop, num_nodes = config["max_hop"], config["num_nodes"]
attn_num_layers, attn_num_heads = config["attn_num_layers"], config["attn_num_heads"]

class PathAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(PathAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, feat, padding_mask):
        """
        Args:
            feat: (L, N, E) where L = S is the target sequence length, N is the batch size, E is the embedding dimension
            padding_mask: (N, S)` where N is the batch size, S is the source sequence length.
        """
        attn_output, attn_output_weights = self.attention(feat, feat, feat, key_padding_mask=padding_mask)
        # feat = feat + attn_output
        return attn_output

class LinearBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(c_in, c_out, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class MHScoreLayer(nn.Module):
    def __init__(self, max_path_len, input_dim, node_num, pool_layer=nn.MaxPool1d):
        super(MHScoreLayer, self).__init__()
        self.max_path_len = max_path_len
        self.input_dim = input_dim
        self.node_num = node_num
        self.hop_max_pooling = pool_layer(kernel_size=self.max_path_len)
        self.fcb = nn.Sequential(
            LinearBlock(self.input_dim, self.input_dim // 2),
            LinearBlock(self.input_dim // 2, self.input_dim // 2)
        )
        self.fc = nn.Linear(self.input_dim // 2, 1)

    def forward(self, feat, padding_mask, fusion_type='maxpooling'):
        """

        Args:
            feat: [L, B, D]
            padding_mask: [B, L]
            fusion_type:
        """
        feat = feat.permute(1, 2, 0)  # (max_path_len, L, ke_dim) -> (L, ke_dim, max_path_len)
        # (L, max_path) -> (L, max_path, 1) -> (L, max_path, ke_dim) -> (L, ke_dim, max_path)
        padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, self.input_dim).permute(0, 2, 1)
        # valued_mask = 1 - padding_mask
        # weight_mask = feat.sum(1).unsequence(1).repeat(1, self.input_dim, 1)  # B, D, L
        feat = feat.masked_fill(padding_mask, float('-inf'))        # 掩码

        if fusion_type == 'maxpooling':
            feat = self.hop_max_pooling(feat).squeeze(-1)   # (L, ke_dim, max_path) -> (L, ke_dim, 1) -> (L, ke_dim)
        # elif fusion_type == 'attention':
        #     # 将 attention 的输出，加权，
        #     feat = feat * weight_mask


        feat = self.fcb(feat)       # (L, ke_dim) -> (L, ke_dim // 2)
        score = self.fc(feat)       # (L, ke_dim // 2) -> (L, 1)
        return score

def process_feats(origin_path_feats):
    """
    returns:
        mh_feat: (L, NxN, D)
        mh_value_mask: (N, N) -> 1 表示有意义的位置
        mh_padding_mask: (NxN, L) -> 1 表示被 padding 的位置
        max_path_len: int
    """
    max_path_len = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if origin_path_feats[i][j] is not None:
                max_path_len = max(max_path_len, origin_path_feats[i][j].shape[0] // ke_dim)
    padding_feat, padding_mask, i1d_to_ij2d, value_mask = [], [], {}, torch.zeros((num_nodes, num_nodes))
    cur_i1d = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if origin_path_feats[i][j] is not None:
                zero_feat = torch.zeros((1, max_path_len, ke_dim))         # (1, max_path_len, ke_dim)
                one_mask = torch.ones(1, max_path_len)                          # (1, max_path_len)
                value_feat = origin_path_feats[i][j].view(-1, ke_dim)      # (path_len, ke_dim)
                zero_feat[:, :value_feat.shape[0]] = value_feat                 # (1, max_path_len, ke_dim) 填充路径，其余位置用0填充
                one_mask[:, :value_feat.shape[0]] = False                       # (1, max_path_len) 填充路径，其余位置用1填充
                padding_feat.append(zero_feat)                                  # L * (1, max_path_len, ke_dim)
                padding_mask.append(one_mask)                                   # L * (1, max_path_len)
                value_mask[i][j] = True                                         # 将有路径的结点对置为 1 (num_nodes, num_nodes)
                i1d_to_ij2d[cur_i1d] = [i, j]                                   # 字典存储有路径的结点对
                cur_i1d += 1
    mh_feat = torch.cat(padding_feat, dim=0).permute(1, 0, 2).to(device)     # (L, max_path_len, ke_dim) -> (max_path_len, L, ke_dim)
    mh_padding_mask = torch.cat(padding_mask, dim=0).to(device).bool()             # (L, max_path_len)
    mh_value_mask = value_mask.to(device).bool()                                   # (num_nodes, num_nodes)
    return max_path_len, mh_feat, mh_value_mask, mh_padding_mask, i1d_to_ij2d

def main():
    path_feat_prefix = './{}/'.format(ke_model)
    # process path feat
    path_feat_path = path_feat_prefix + '{}_{}hop.pkl'.format(ke_dim, max_hop)
    with open(path_feat_path, mode='rb') as f:
        origin_path_feats = pickle.load(f)
    # after path feature fusion
    max_path_len, mh_feat, mh_value_mask, mh_padding_mask, i1d_to_ij2d = \
        process_feats(origin_path_feats)  # 注意，value_mask与padding_mask的意义相反
    for att_layer in range(attn_num_layers):
        mh_feat_encoder = PathAttention(embedding_dim=ke_dim, num_heads=attn_num_heads).to(device)
        att_out = mh_feat_encoder(mh_feat, mh_padding_mask)
        mh_feat = mh_feat + att_out
    mh_score_layer = MHScoreLayer(max_path_len, ke_dim, num_nodes).to(device)
    score = mh_score_layer(mh_feat, mh_padding_mask)  # (L, 1)
    mh_score = torch.zeros((num_nodes, num_nodes))
    for i1d, (i2d, j2d) in i1d_to_ij2d.items():
        mh_score[i2d][j2d] = score[i1d]
    np.savetxt(path_feat_prefix + '{}_{}hop.csv'.format(ke_dim, max_hop), mh_score.detach().cpu().numpy())

if __name__ == '__main__':
    main()