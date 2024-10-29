# we add the functions that we need to process the data
import osmnx as ox
import requests
from scipy.spatial import cKDTree
import pandas as pd
import networkx as nx
import numpy as np

# def travel_time(x, y, sub_pcs):
#     target_coords = [x, y]
#     candidate_pcs = sub_pcs['facility_id'].to_list()
#     coords = [[x,y] for x,y in zip(sub_pcs.facility_x.to_list(),sub_pcs.facility_y.to_list())]
#     t_i = dict()
#     for idx,i in enumerate(coords):
#         olon,olat = target_coords
#         dlon,dlat = i
#         url = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}".format(olon,olat,dlon,dlat)
#         r = requests.get(url)
#         res = r.json()
#         t_i[idx] = round(res['routes'][0]['duration']/60,2)
#     return candidate_pcs, t_i

def assign_close_pcs(x, y, pcs_gdf):
    pcs_tree = cKDTree(pcs_gdf[['facility_x', 'facility_y']])
    _, nearest_indices = pcs_tree.query([x, y], k=3)
    sub_pcs = pcs_gdf.iloc[nearest_indices]
    # candidate_pcs, t_i = travel_time(x, y, sub_pcs)
    # landfill_id = min(t_i, key=t_i.get)
    return sub_pcs # candidate_pcs[landfill_id]

def od_dataframe(cbg_gdf, pcs_gdf):
    # iterate through each cbg and find the closest pcs and create a dataframe
    for idx, row in cbg_gdf.iterrows():
        sub_pcs = assign_close_pcs(row['ct_x'], row['ct_y'], pcs_gdf)
        sub_pcs['GEOID'] = row['GEOID']
        if idx == 0:
            pcs_cbg_od = sub_pcs
        else:
            pcs_cbg_od = pd.concat([pcs_cbg_od, sub_pcs])
    pcs_cbg_od = pcs_cbg_od.merge(cbg_gdf[['GEOID', 'ct_x', 'ct_y']], on='GEOID', how='left')
    pcs_cbg_od.reset_index(inplace=True, drop=True)
    return pcs_cbg_od

def generate_od_df(cbg_gdf, pcs_gdf):
    # cbg_gdf['ct_x'] = cbg_gdf.geometry.centroid.x
    # cbg_gdf['ct_y'] = cbg_gdf.geometry.centroid.y
    cbg_gdf = cbg_gdf[['GEOID', 'ct_x', 'ct_y']]
    # pcs_gdf['facility_x'] = pcs_gdf.geometry.centroid.x
    # pcs_gdf['facility_y'] = pcs_gdf.geometry.centroid.y
    pcs_gdf = pcs_gdf[['facility_id', 'facility_x', 'facility_y']]
    pcs_cbg_od = od_dataframe(cbg_gdf, pcs_gdf)
    return pcs_cbg_od

# function that gets a point as the input and returns the closest node 
def closest_node(od_df, nodes_gdf):
    node_tree = cKDTree(np.c_[nodes_gdf['geometry'].x, nodes_gdf['geometry'].y])
    _,osm_indices = node_tree.query(np.c_[od_df.ct_x, od_df.ct_y])
    _,osm_indices2 = node_tree.query(np.c_[od_df.facility_x, od_df.facility_y])
    od_df['closest_origin_node'] = osm_indices # nodes_gdf['geometry'].iloc[osm_indices]['osm_id'].values
    od_df['closest_destination_node'] = osm_indices2 # nodes_gdf['geometry'].iloc[osm_indices2]['osm_id'].values
    return od_df

def compute_travel_time(od_matrix, graph):
    try:
        # Use networkx's Dijkstra's algorithm to find the shortest path
        origin_node = od_matrix['closest_origin_node'] #.values[0]
        destination_node = od_matrix['closest_destination_node'] #.values[0]
        route = nx.dijkstra_path(graph, origin_node, destination_node, weight='travel_time')
        # Compute the total travel time
        travel_time = sum(graph[u][v][0]['travel_time'] for u, v in zip(route[:-1], route[1:]))
        return travel_time
    except nx.NetworkXNoPath:
        return float('inf')  # No path found, return infinity
    except Exception as e:
        return np.nan
    
def osm_processing(socio_gdf, afc_shp, G):
    socio_gdf_sub = socio_gdf[['GEOID', 'ct_x', 'ct_y']]
    afc_shp_sub = afc_shp[['facility_id', 'facility_x', 'facility_y']]
    od_marix = generate_od_df(socio_gdf_sub, afc_shp_sub)

    # G = ox.load_graphml(filepath = "D:/Users/hgazmeh/Codes/pfas/us_roadnetwork/us_interstate_road_network_graph.grpahml")
    gnodes = ox.graph_to_gdfs(G, edges=False)
    gnodes.to_crs(epsg=4269, inplace=True)
    gnodes['x'] = gnodes['geometry'].x
    gnodes['y'] = gnodes['geometry'].y
    gnodes['osm_id'] = gnodes.index
    od_marix = closest_node(od_marix, gnodes)
    od_marix['travel_time_sec'] = od_marix.apply(lambda row: compute_travel_time(row, G), axis=1)
    # od_marix.to_csv('selected_metros/{}/{}_od_matrix_with_travel_time.csv'.format(metro_name, metro_name), index=False)
    return od_marix