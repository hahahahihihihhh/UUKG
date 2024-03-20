import pandas as pd

entity2id_NYC = pd.read_csv(".././data/NYC/entity2id_NYC.txt", sep = " ", header = None, names = ['entity','id'])

def used_area_KG_id():
    used_area = pd.read_csv("used_entity/NYC_used_area.csv")
    area_KG_id = pd.read_csv("xxx_KG_id/NYC_area_KG_id.csv")
    dict_area_KG_id = {}
    for _area_id, _KG in area_KG_id.values:
        dict_area_KG_id[_area_id] = _KG
    used_area_KG_id = []
    for _i, _area in enumerate(used_area.values.flatten()):
        used_area_KG_id.append([_area, dict_area_KG_id[_area]])
    pd.DataFrame(data=used_area_KG_id, columns=['area_id', 'KG']).to_csv('used_xxx_KG_id/NYC_used_area_KG_id.csv', index=False)

def used_road_KG_id():
    used_road = pd.read_csv("used_entity/NYC_used_road.csv")
    road_KG_id = pd.read_csv("xxx_KG_id/NYC_road_KG_id.csv")
    dict_road_KG_id = {}
    for _road_id, _KG in road_KG_id.values:
        dict_road_KG_id[_road_id] = _KG
    used_road_KG_id = []
    for _i, _road in enumerate(used_road.values.flatten()):
        used_road_KG_id.append([_road, dict_road_KG_id[_road]])
    pd.DataFrame(data=used_road_KG_id, columns=['road_id', 'KG']).to_csv('used_xxx_KG_id/NYC_used_road_KG_id.csv', index=False)

def used_POI_KG_id():
    used_POI = pd.read_csv("used_entity/NYC_used_POI.csv")
    POI_KG_id = pd.read_csv("xxx_KG_id/NYC_POI_KG_id.csv")
    dict_POI_KG_id = {}
    for _POI_id, _KG in POI_KG_id.values:
        dict_POI_KG_id[_POI_id] = _KG
    used_POI_KG_id = []
    for _i, _POI in enumerate(used_POI.values.flatten()):
        used_POI_KG_id.append([_POI, dict_POI_KG_id[_POI]])
    pd.DataFrame(data=used_POI_KG_id, columns=['POI_id', 'KG']).to_csv('used_xxx_KG_id/NYC_used_POI_KG_id.csv', index=False)

if __name__ == "__main__":
    used_area_KG_id()
    used_road_KG_id()
    used_POI_KG_id()