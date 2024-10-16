"""
borough, area, road, RC, junction, JC, poi, PC
"""

with open("../UrbanKG/CHI/entity2id_CHI.txt") as f:
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

print("CHI Borough: ", len(Borough))
print("CHI Area: ", len(Area))
print("CHI Road: ", len(Road))
print("CHI RC: ", len(RC))
print("CHI Junction: ", len(Junction))
print("CHI JC: ", len(JC))
print("CHI POI: ", len(POI))
print("CHI PC: ", len(PC))
print("---------------------------------------------------------")
"""

"""
Ent = []
Rel = []
Triplet = []
Train = []
Valid = []
Test = []

with open("../UrbanKG/CHI/entity2id_CHI.txt") as f:
    for line in f.readlines():
        Ent.append(line)

print("CHI Ent:", len(Ent))

with open("../UrbanKG/CHI/relation2id_CHI.txt") as f:
    for line in f.readlines():
        Rel.append(line)
print("CHI Rel:", len(Rel))

with open("../UrbanKG/CHI/triplets_CHI.txt") as f:
    for line in f.readlines():
        Triplet.append(line)
print("CHI Triplet:", len(Triplet))

with open("../UrbanKG/CHI/train_CHI.txt") as f:
    for line in f.readlines():
        Train.append(line)
print("CHI Train:", len(Train))

with open("../UrbanKG/CHI/valid_CHI.txt") as f:
    for line in f.readlines():
        Valid.append(line)
print("CHI Valid:", len(Valid))

with open("../UrbanKG/CHI/test_CHI.txt") as f:
    for line in f.readlines():
        Test.append(line)
print("CHI Test:", len(Test))
print("---------------------------------------------------------")
"""

"""
rel_count = {
    "PLA": 0,
    "RLA": 0,
    "JLA": 0,
    "PBB": 0,
    "RBB": 0,
    "JBB": 0,
    "ALB": 0,
    "JBR": 0,
    "BNB": 0,
    "ANA": 0,
    "PHPC": 0,
    "RHRC": 0,
    "JHJC": 0
}
with open("../UrbanKG/CHI/UrbanKG_CHI.txt") as f:
    for line in f.readlines():
        for k, _ in rel_count.items():
            if k in line:
                rel_count[k] += 1
for k, v in rel_count.items():
    print("CHI {}: {}".format(k, v))


'''
CHI:
    CHI Borough:  6
    CHI Area:  77
    CHI Road:  71578
    CHI RC:  6
    CHI Junction:  37342
    CHI JC:  5
    CHI POI:  31573
    CHI PC:  15
    ---------------------------------------------------------
    CHI Ent: 140602
    CHI Rel: 13
    CHI Triplet: 564400
    CHI Train: 507960
    CHI Valid: 28220
    CHI Test: 28220
    ---------------------------------------------------------
    CHI PLA: 31573
    CHI RLA: 71578
    CHI JLA: 37086
    CHI PBB: 31573
    CHI RBB: 71578
    CHI JBB: 37086
    CHI ALB: 123
    CHI JBR: 143156
    CHI BNB: 16
    CHI ANA: 394
    CHI PHPC: 31573
    CHI RHRC: 71578
    CHI JHJC: 37086
'''




