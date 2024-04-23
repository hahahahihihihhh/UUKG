import os
import pandas as pd

dataset = "CHI" # CHI or NYC
entity2id = pd.read_csv(".././data/{0}/entity2id_{0}.txt".format(dataset), sep = " ", header = None, names = ['entity','id'])
xxx_id2KG_id_prefix = "xxx_id2KG_id"

def area_id2KG_id():
    area_id2KG_id = list()
    for _entity, _id in entity2id.values:
        if (_entity.startswith("Area")):
            area_id2KG_id.append([int(_entity[_entity.index("/") + 1: ]), int(_id)])
    (pd.DataFrame(data = area_id2KG_id, columns = ['area_id', 'KG'])
     .to_csv(os.path.join(xxx_id2KG_id_prefix, dataset, 'area_id2KG_id.csv'), index = False))

def road_id2KG_id():
    road_id2KG_id = list()
    for _entity, _id in entity2id.values:
        if (_entity.startswith("Road")):
            road_id2KG_id.append([int(_entity[_entity.index("/") + 1:]), int(_id)])
    (pd.DataFrame(data=road_id2KG_id, columns=['road_id', 'KG'])
     .to_csv(os.path.join(xxx_id2KG_id_prefix, dataset, 'road_id2KG_id.csv'), index=False))

def POI_id2KG_id():
    POI_id2KG_id = list()
    for _entity, _id in entity2id.values:
        if (_entity.startswith("POI")):
            POI_id2KG_id.append([int(_entity[_entity.index("/") + 1:]), int(_id)])
    (pd.DataFrame(data=POI_id2KG_id, columns=['POI_id', 'KG'])
     .to_csv(os.path.join(xxx_id2KG_id_prefix, dataset, 'POI_id2KG_id.csv'), index=False))

if __name__ == "__main__":
    area_id2KG_id()
    road_id2KG_id()
    POI_id2KG_id()