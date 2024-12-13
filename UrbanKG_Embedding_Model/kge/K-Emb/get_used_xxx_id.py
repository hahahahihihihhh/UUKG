import os.path
import pandas as pd

dataset = "CHI" # CHI or NYC
used_xxx_id_prefix = "used_xxx_id"
used_xxx_prefix_path = '../../../USTP_Model/raw_data/'

def get_used_area_id():
    # taxi, 311Service, crime
    used_area = pd.read_csv(used_xxx_prefix_path + '{0}Taxi{1}/{0}Taxi{1}.geo'
                            .format(dataset, "20190406" if dataset == "CHI" else "20200406"))

    (pd.DataFrame(data=list(used_area['geo_id']), columns=["used_area_id"], dtype=int)
     .to_csv(os.path.join(used_xxx_id_prefix, dataset, 'used_area_id.csv'), index=False))
def get_used_road_id():
    # bike
    used_road = pd.read_csv(used_xxx_prefix_path + '{0}Bike{1}/{0}Bike{1}.geo'
                            .format(dataset, "20190406" if dataset == "CHI" else "20200406"))
    (pd.DataFrame(data=list(used_road['geo_id']), columns=["used_road_id"], dtype=int)
     .to_csv(os.path.join(used_xxx_id_prefix, dataset, 'used_road_id.csv'), index=False))
def get_used_POI_id():
    # human
    used_POI = pd.read_csv(used_xxx_prefix_path + '{0}Human{1}/{0}Human{1}.geo'
                            .format(dataset, "20190406" if dataset == "CHI" else "20200406"))
    (pd.DataFrame(data=list(used_POI['geo_id']), columns=["used_POI_id"], dtype=int)
     .to_csv(os.path.join(used_xxx_id_prefix, dataset, 'used_POI_id.csv'), index=False))

if __name__ == "__main__":
    get_used_area_id()
    get_used_road_id()
    get_used_POI_id()

