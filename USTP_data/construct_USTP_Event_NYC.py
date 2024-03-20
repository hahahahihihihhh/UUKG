"""

trafficstatepoin The spatial dimension is one-dimensional datasets (i.e. point-based/segment-based/area-based datasets)

The processed files are stored in ./USTP_Model
File format after alignment, filtering and preprocessing:
    event dataset
   .geo：geo_id,type,coordinates
   .grid：dyna_id,type,time,row_id,column_id,flow
   .rel：rel_id,type,origin_id,destination_id,cost

"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point

dataframe2 = gpd.read_file('../UrbanKG_data/Meta_data/NYC/Administrative_data/Area/Area.shp')
# Convert to latitude-longitude coordinate system
dataframe2 = dataframe2.to_crs('EPSG:4326')
seleceted_colums2 = ['OBJECTID', 'zone', 'geometry']
area_dataframe = dataframe2[seleceted_colums2]
## filter
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 1]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 103]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 104]


"""

USTP_data 4: crime

"""
##################################################################################################
##################################################################################################

crime_dataframe = pd.read_csv('./Processed_data/NYC/NYC_crime.csv')
crime_dataframe['time'] = pd.to_datetime(crime_dataframe['time'], format="%Y/%m/%d %H:%M")
crime_dataframe['time'] = crime_dataframe['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
crime_numpy = crime_dataframe[['time', 'area_id']].values

begin_time = '2021-01-01 00:00:00'
format_pattern = '%Y-%m-%d %H:%M:%S'

crime_flow = np.zeros([4380, 263])

for i in range(crime_numpy.shape[0]):
    time_spannn_out = datetime.strptime(str(crime_numpy[i][0]), format_pattern) - datetime.strptime(begin_time, format_pattern)
    total_seconds_out = int(time_spannn_out.total_seconds())
    time_step_out =  int(total_seconds_out / 7200)
    if 0<= time_step_out < 4380 :
        crime_flow[time_step_out][ int(crime_numpy[i][1]) -1 ] = 1

now = datetime.strptime(begin_time, format_pattern)

NYCCrime20210112_dyna = []
NYCCrime20210112_dyna.append('dyna_id,type,time,entity_id,flow')
NYCCrime20210112_geo = []
NYCCrime20210112_geo.append('geo_id,type,coordinates')

dyna_id = 0
type = 'state'

for i in tqdm(range(crime_flow.shape[1])):
    if i != 0 and i != 102 and i != 103:
        for j in range(crime_flow.shape[0]):
            time_origin = now + timedelta(minutes=120 * j)
            time_write = time_origin.strftime('%Y-%m-%dT%H:%M:%SZ')
            flow = crime_flow[j][i]

            grid_record = str(dyna_id) + ',' + type + ',' + str(time_write) + ',' + str(i+1) + ',' + str(int(flow))
            dyna_id += 1
            NYCCrime20210112_dyna.append(grid_record)

        geo_record = str(i+1) + ',' + 'Point' + ',"[]"'
        NYCCrime20210112_geo.append(geo_record)

with open(r'./USTP/NYC/NYCCrime20210112/NYCCrime20210112.dyna','w') as f1:
    for i in range(len(NYCCrime20210112_dyna)):
        f1.write(NYCCrime20210112_dyna[i])
        f1.write('\n')
f1.close()

with open(r'./USTP/NYC/NYCCrime20210112/NYCCrime20210112.geo','w') as f1:
    for i in range(len(NYCCrime20210112_geo)):
        f1.write(NYCCrime20210112_geo[i])
        f1.write('\n')
f1.close()

NYCCrime20210112_rel = []
NYCCrime20210112_rel.append('rel_id,type,origin_id,destination_id,cost')

rel_id = 0
type = 'geo'
for i in tqdm(range(area_dataframe.shape[0])):
    head_area = area_dataframe.iloc[i].geometry
    for j in range(area_dataframe.shape[0]):
        tail_area = area_dataframe.iloc[j].geometry
        distance = head_area.distance(tail_area)
        NYCCrime20210112_rel.append(str(rel_id) + ',' +  type + ',' + str(area_dataframe.iloc[i].OBJECTID) + ',' + str(area_dataframe.iloc[j].OBJECTID)
                                   + ',' + str(distance))
        rel_id += 1

with open(r'./USTP/NYC/NYCCrime20210112/NYCCrime20210112.rel','w') as f1:
    for i in range(len(NYCCrime20210112_rel)):
        f1.write(NYCCrime20210112_rel[i])
        f1.write('\n')
f1.close()

##################################################################################################
##################################################################################################
"""

USTP_data 5: 311 service

"""
service_dataframe = pd.read_csv('./Processed_data/NYC/NYC_311_service.csv')
service_dataframe['time'] = pd.to_datetime(service_dataframe['time'], format="%Y/%m/%d %H:%M")
service_dataframe['time'] = service_dataframe['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
service_numpy = service_dataframe[['time', 'area_id']].values

tmp = sorted(list(set(service_dataframe[['area_id']].values.flatten())))
begin_time = '2021-01-01 00:00:00'
format_pattern = '%Y-%m-%d %H:%M:%S'

service_flow = np.zeros([4380, 263])

for i in range(service_numpy.shape[0]):
    time_spannn_out = datetime.strptime(str(service_numpy[i][0]), format_pattern) - datetime.strptime(begin_time, format_pattern)
    total_seconds_out = int(time_spannn_out.total_seconds())
    time_step_out =  int(total_seconds_out / 7200)
    if 0<= time_step_out < 4380 :
        service_flow[time_step_out][ int(service_numpy[i][1]) -1 ] = 1

now = datetime.strptime(begin_time, format_pattern)

NYCService20210112_dyna = []
NYCService20210112_dyna.append('dyna_id,type,time,entity_id,flow')
NYCService20210112_geo = []
NYCService20210112_geo.append('geo_id,type,coordinates')

dyna_id = 0
type = 'state'

for i in tqdm(range(service_flow.shape[1])):
    if i != 0 and i != 102 and i != 103:
        for j in range(service_flow.shape[0]):
            time_origin = now + timedelta(minutes=120 * j)
            time_write = time_origin.strftime('%Y-%m-%dT%H:%M:%SZ')
            flow = service_flow[j][i]

            grid_record = str(dyna_id) + ',' + type + ',' + str(time_write) + ',' + str(i+1) + ',' + str(int(flow))
            dyna_id += 1
            NYCService20210112_dyna.append(grid_record)

        geo_record = str(i+1) + ',' + 'Point' + ',"[]"'
        NYCService20210112_geo.append(geo_record)

with open(r'./USTP/NYC/NYC311Service20210112/NYC311Service20210112.dyna','w') as f1:
    for i in range(len(NYCService20210112_dyna)):
        f1.write(NYCService20210112_dyna[i])
        f1.write('\n')
f1.close()

with open(r'./USTP/NYC/NYC311Service20210112/NYC311Service20210112.geo','w') as f1:
    for i in range(len(NYCService20210112_geo)):
        f1.write(NYCService20210112_geo[i])
        f1.write('\n')
f1.close()

NYCService20210112_rel = []
NYCService20210112_rel.append('rel_id,type,origin_id,destination_id,cost')

rel_id = 0
type = 'geo'
for i in tqdm(range(area_dataframe.shape[0])):
    head_area = area_dataframe.iloc[i].geometry
    for j in range(area_dataframe.shape[0]):
        tail_area = area_dataframe.iloc[j].geometry
        distance = head_area.distance(tail_area)
        NYCService20210112_rel.append(str(rel_id) + ',' +  type + ',' + str(area_dataframe.iloc[i].OBJECTID) + ',' + str(area_dataframe.iloc[j].OBJECTID)
                                   + ',' + str(distance))
        rel_id += 1

with open(r'./USTP/NYC/NYC311Service20210112/NYC311Service20210112.rel','w') as f1:
    for i in range(len(NYCService20210112_rel)):
        f1.write(NYCService20210112_rel[i])
        f1.write('\n')
f1.close()