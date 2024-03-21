import pandas as pd

entity2id_CHI = pd.read_csv(".././data/CHI/entity2id_CHI.txt", sep = " ", header = None, names = ['entity','id'])

def used_area_id2KG_id():
    used_area_id = pd.read_csv("used_xxx_id/CHI_used_area_id.csv")
    area_id2KG_id = pd.read_csv("xxx_id2KG_id/CHI_area_id2KG_id.csv")
    dict_area_id2KG_id = {}
    for _area_id, _KG_id in area_id2KG_id.values:
        dict_area_id2KG_id[_area_id] = _KG_id
    used_area_id2KG_id = []
    for _i, _area_id in enumerate(used_area_id.values.flatten()):
        used_area_id2KG_id.append([_area_id, dict_area_id2KG_id[_area_id]])
    pd.DataFrame(data=used_area_id2KG_id, columns=['area_id', 'KG_id']).to_csv('used_xxx_id2KG_id/CHI_used_area_id2KG_id.csv', index=False)

def used_road_id2KG_id():
    used_road_id = pd.read_csv("used_xxx_id/CHI_used_road_id.csv")
    road_id2KG_id = pd.read_csv("xxx_id2KG_id/CHI_road_id2KG_id.csv")
    dict_road_id2KG_id = {}
    for _road_id, _KG_id in road_id2KG_id.values:
        dict_road_id2KG_id[_road_id] = _KG_id
    used_road_id2KG_id = []
    for _i, _road_id in enumerate(used_road_id.values.flatten()):
        used_road_id2KG_id.append([_road_id, dict_road_id2KG_id[_road_id]])
    pd.DataFrame(data=used_road_id2KG_id, columns=['road_id', 'KG_id']).to_csv('used_xxx_id2KG_id/CHI_used_road_id2KG_id.csv', index=False)

def used_POI_id2KG_id():
    used_POI_id = pd.read_csv("used_xxx_id/CHI_used_POI_id.csv")
    POI_id2KG_id = pd.read_csv("xxx_id2KG_id/CHI_POI_id2KG_id.csv")
    dict_POI_id2KG_id = {}
    for _POI_id, _KG_id in POI_id2KG_id.values:
        dict_POI_id2KG_id[_POI_id] = _KG_id
    used_POI_id2KG_id = []
    for _i, _POI_id in enumerate(used_POI_id.values.flatten()):
        used_POI_id2KG_id.append([_POI_id, dict_POI_id2KG_id[_POI_id]])
    pd.DataFrame(data=used_POI_id2KG_id, columns=['POI_id', 'KG_id']).to_csv('used_xxx_id2KG_id/CHI_used_POI_id2KG_id.csv', index=False)

if __name__ == "__main__":
    used_area_id2KG_id()
    used_road_id2KG_id()
    used_POI_id2KG_id()