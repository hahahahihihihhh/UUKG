import pandas as pd

def get_used_area_id():
    # taxi, 311Service, crime
    used_area = pd.read_csv('../raw_data/CHITaxi20190406/CHITaxi20190406.geo')
    pd.DataFrame(data=list(used_area['geo_id']), columns=["used_area_id"], dtype=int).to_csv('used_xxx_id/CHI_used_area_id.csv'
                                                                                  , index=False)

def get_used_road_id():
    # bike
    used_road = pd.read_csv('../raw_data/CHIBike20190406/CHIBike20190406.geo')
    pd.DataFrame(data=list(used_road['geo_id']), columns=["used_road_id"], dtype=int).to_csv('used_xxx_id/CHI_used_road_id.csv',
                                                                                  index=False)
def get_used_POI_id():
    # human
    used_POI = pd.read_csv('../raw_data/CHIHuman20190406/CHIHuman20190406.geo')
    pd.DataFrame(data=list(used_POI['geo_id']), columns=["used_POI_id"], dtype=int).to_csv('used_xxx_id/CHI_used_POI_id.csv',
                                                                                  index=False)
if __name__ == "__main__":
    get_used_area_id()
    get_used_road_id()
    get_used_POI_id()
