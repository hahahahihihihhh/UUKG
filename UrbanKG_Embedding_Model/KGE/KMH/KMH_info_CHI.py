import os.path

import numpy as np
import pickle
import torch

ANA, area_num = 1, 77
embeddings_prefix = "../xxx_embeddings/CHI/GIE"
KMHpath_prefix = "CHI/GIE"
def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_obj(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

def main():
    area_graph = load_obj("area_graph_CHI.pkl")
    area_embedding = np.load(os.path.join(embeddings_prefix, "area_embeddings.npy"))
    rel_embedding =  np.load(os.path.join(embeddings_prefix, "rel_embeddings.npy"))
    path = [[None for _j in range(area_num)] for _i in range(area_num)]
    for _s in range(area_num):
        for _t in range(area_num):
            p = None
            if (area_graph[_s][_t] == 1):
                p = torch.tensor(np.concatenate([area_embedding[_s],
                                    rel_embedding[ANA],
                                    area_embedding[_t]], axis=0))
                assert _s != _t
            elif (_s == _t):
                p = torch.tensor(area_embedding[_s])
            path[_s][_t] = p
    save_obj(os.path.join(KMHpath_prefix, '{}_1hop.pkl'.format(area_embedding.shape[1])), path)

if __name__ == '__main__':
    main()
