import os.path
from idlelib.iomenu import encoding

import numpy as np
import pickle
import torch

ANA, area_num = 1, 77
ke_dim = 32
embeddings_prefix = "../xxx_embeddings/CHI/TransE"
KMHpath_prefix = "./"
def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_obj(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    area_graph = load_obj("area_graph_CHI.pkl")
    area_embedding = np.load(os.path.join(embeddings_prefix, "area_{}d.npy".format(ke_dim)))
    rel_embedding =  np.load(os.path.join(embeddings_prefix, "rel_{}d.npy".format(ke_dim)))
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
    ensure_dir(KMHpath_prefix)
    save_obj(KMHpath_prefix + '{}_1hop.pkl'.format(ke_dim), path)

if __name__ == '__main__':
    main()
