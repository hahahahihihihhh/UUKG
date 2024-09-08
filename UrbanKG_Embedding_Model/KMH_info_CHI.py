import os.path

import numpy as np
import pickle

ANA, area_num = 1, 77
embeddings_prefix = "KGE/xxx_embeddings/CHI/GIE"
KMHpath_prefix = "KGE/KMH/CHI/GIE"
def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_obj(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

def main():
    area_graph = load_obj("KGE/KMH/area_graph_CHI.pkl")
    area_embedding = np.load(os.path.join(embeddings_prefix, "area_embeddings.npy"))
    rel_embedding =  np.load(os.path.join(embeddings_prefix, "rel_embeddings.npy"))
    path = [[np.array for _j in range(area_num)] for _i in range(area_num)]
    for _s in range(area_num):
        for _t in range(area_num):
            p = np.array([])
            if (area_graph[_s][_t] == 1):
                p = np.concatenate([area_embedding[_s].reshape(1, area_embedding[_s].shape[0]),
                                    rel_embedding[ANA].reshape(1, rel_embedding[ANA].shape[0]),
                                    area_embedding[_t].reshape(1, area_embedding[_t].shape[0])], axis=1)
                assert _s != _t
            elif (_s == _t):
                p = area_embedding[_s].reshape(1, area_embedding[_s].shape[0])
            path[_s][_t] = p
    save_obj(os.path.join(KMHpath_prefix, 'one_hop.pkl'), path)

if __name__ == '__main__':
    main()
