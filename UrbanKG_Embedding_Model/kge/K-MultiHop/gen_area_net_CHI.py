import pickle
import pandas as pd

ANA, area_num = 1, 77
df = pd.read_csv("../../data/CHI/triplets_CHI.txt", sep=" ", header=None)
entity2id = pd.read_csv("../../data/CHI/entity2id_CHI.txt", sep=" ", header=None)

def get_area_id(area: str) -> int:
    return int(area.split("/")[1])

def save_obj(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

def main():
    id2entity = {}
    for _e, _i in entity2id.values:
        id2entity[_i] = _e
    area_graph = [[0 for _i in range(area_num)] for _ in range(area_num)]
    for _h, _r, _t in df.values:
        if _r == ANA:
            _h, _t = get_area_id(id2entity[_h]), get_area_id(id2entity[_t])
            area_graph[_h - 1][_t - 1] = 1
    save_obj('area_graph_CHI.pkl', area_graph)

if __name__ == '__main__':
    main()