import numpy as np
import pandas as pd

entity2id_NYC = pd.read_csv(".././data/NYC/entity2id_NYC.txt", sep = " ", header = None, names = ['entity','id'])

def area_KG_id():
    area_KG_id = list()
    for _entity, _id in entity2id_NYC.values:
        if (_entity.startswith("Area")):
            area_KG_id.append([int(_entity[_entity.index("/") + 1: ]), int(_id)])
    pd.DataFrame(data = area_KG_id, columns = ['area_id', 'KG']).to_csv('./xxx_KG_id/NYC_area_KG_id.csv', index = False)

def road_KG_id():
    road_KG_id = list()
    for _entity, _id in entity2id_NYC.values:
        if (_entity.startswith("Road")):
            road_KG_id.append([int(_entity[_entity.index("/") + 1:]), int(_id)])
    pd.DataFrame(data=road_KG_id, columns=['road_id', 'KG']).to_csv('./xxx_KG_id/NYC_road_KG_id.csv', index=False)

def POI_KG_id():
    POI_KG_id = list()
    for _entity, _id in entity2id_NYC.values:
        if (_entity.startswith("POI")):
            POI_KG_id.append([int(_entity[_entity.index("/") + 1:]), int(_id)])
    pd.DataFrame(data=POI_KG_id, columns=['POI_id', 'KG']).to_csv('./xxx_KG_id/NYC_POI_KG_id.csv', index=False)

if __name__ == "__main__":
    area_KG_id()
    road_KG_id()
    POI_KG_id()