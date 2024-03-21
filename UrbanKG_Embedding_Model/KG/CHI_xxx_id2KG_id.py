import pandas as pd

entity2id_CHI = pd.read_csv(".././data/CHI/entity2id_CHI.txt", sep = " ", header = None, names = ['entity','id'])

def area_id2KG_id():
    area_id2KG_id = list()
    for _entity, _id in entity2id_CHI.values:
        if (_entity.startswith("Area")):
            area_id2KG_id.append([int(_entity[_entity.index("/") + 1: ]), int(_id)])
    pd.DataFrame(data = area_id2KG_id, columns = ['area_id', 'KG']).to_csv('./xxx_id2KG_id/CHI_area_id2KG_id.csv', index = False)

def road_id2KG_id():
    road_id2KG_id = list()
    for _entity, _id in entity2id_CHI.values:
        if (_entity.startswith("Road")):
            road_id2KG_id.append([int(_entity[_entity.index("/") + 1:]), int(_id)])
    pd.DataFrame(data=road_id2KG_id, columns=['road_id', 'KG']).to_csv('./xxx_id2KG_id/CHI_road_id2KG_id.csv', index=False)

def POI_id2KG_id():
    POI_id2KG_id = list()
    for _entity, _id in entity2id_CHI.values:
        if (_entity.startswith("POI")):
            POI_id2KG_id.append([int(_entity[_entity.index("/") + 1:]), int(_id)])
    pd.DataFrame(data=POI_id2KG_id, columns=['POI_id', 'KG']).to_csv('./xxx_id2KG_id/CHI_POI_id2KG_id.csv', index=False)

if __name__ == "__main__":
    area_id2KG_id()
    road_id2KG_id()
    POI_id2KG_id()