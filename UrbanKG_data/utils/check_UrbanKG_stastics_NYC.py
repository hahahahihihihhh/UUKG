"""
borough, area, road, RC, junction, JC, poi, PC
"""

with open("../UrbanKG/NYC/entity2id_NYC.txt") as f:
    Borough = []
    Area = []
    Road = []
    RC = []
    Junction = []
    JC = []
    POI = []
    PC = []
    for line in f.readlines():
        if 'Borough/' in line:
            Borough.append(line)
        if 'Area/' in line:
            Area.append(line)
        if 'Road/' in line:
            Road.append(line)
        if 'RC/' in line:
            RC.append(line)
        if 'Junction/' in line:
            Junction.append(line)
        if 'JC/' in line:
            JC.append(line)
        if 'POI/' in line:
            POI.append(line)
        if 'PC/' in line:
            PC.append(line)

print("NYC Borough: ", len(Borough))
print("NYC Area: ", len(Area))
print("NYC Road: ", len(Road))
print("NYC RC: ", len(RC))
print("NYC Junction: ", len(Junction))
print("NYC JC: ", len(JC))
print("NYC POI: ", len(POI))
print("NYC PC: ", len(PC))
print("---------------------------------------------------------")
"""

"""
Ent = []
Rel = []
Triplet = []
Train = []
Valid = []
Test = []

with open("../UrbanKG/NYC/entity2id_NYC.txt") as f:
    for line in f.readlines():
        Ent.append(line)

print("NYC Ent:", len(Ent))

with open("../UrbanKG/NYC/relation2id_NYC.txt") as f:
    for line in f.readlines():
        Rel.append(line)
print("NYC Rel:", len(Rel))

with open("../UrbanKG/NYC/triplets_NYC.txt") as f:
    for line in f.readlines():
        Triplet.append(line)
print("NYC Triplet:", len(Triplet))

with open("../UrbanKG/NYC/train_NYC.txt") as f:
    for line in f.readlines():
        Train.append(line)
print("NYC Train:", len(Train))

with open("../UrbanKG/NYC/valid_NYC.txt") as f:
    for line in f.readlines():
        Valid.append(line)
print("NYC Valid:", len(Valid))

with open("../UrbanKG/NYC/test_NYC.txt") as f:
    for line in f.readlines():
        Test.append(line)
print("NYC Test:", len(Test))


'''
NYC:
    NYC Borough:  5
    NYC Area:  260
    NYC Road:  110919
    NYC RC:  6
    NYC Junction:  62627
    NYC JC:  5
    NYC POI:  62450
    NYC PC:  15
    ---------------------------------------------------------
    NYC Ent: 236287
    NYC Rel: 13
    NYC Triplet: 930240
    NYC Train: 837216
    NYC Valid: 46512
    NYC Test: 46512
'''



