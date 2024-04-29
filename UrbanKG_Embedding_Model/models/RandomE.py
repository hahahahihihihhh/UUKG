import torch

from models.base import KGModel

RANDOM_MODELS = ["RandomE"]

class RandomE(KGModel):

    def __init__(self, args):
        super(RandomE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        #   self.sizes:(14602, 26, 14602)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank),
                                                               dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_queries(self, queries):
        lhs_e = self.entity(queries[:, 0])
        lhs_bias = self.bh(queries[:, 0])
        return lhs_e, lhs_bias

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            rhs_e = self.entity.weight
            rhs_bias = self.bt.weight
            return rhs_e, rhs_bias
        else:
            rhs_e = self.entity(queries[:, 2])
            rhs_bias = self.bh(queries[:, 2])
            return rhs_e, rhs_bias

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if eval_mode:
            score = lhs_e @ rhs_e.transpose(0, 1)
        else:
            score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        return score
