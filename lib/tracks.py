# -*- coding: utf-8 -*-

#  Copyright (C) 2019-2021 Adrian Wöltche
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program. If not, see https://www.gnu.org/licenses/.

import os
import collections
import itertools
import math
import statistics
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely as shp
import matplotlib.pyplot as plt

import lib.geodata

class Tracks(lib.geodata.GeoData):

    def __init__(self, file=None, delimiter=',', crs="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
                 lon="lon", lat="lat", timestamp="timestamp", groupby=None,
                 reproject_crs="+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
                 dump=os.path.join("data", "tracks.pickle"), points=None, reload=False):
        super().__init__()

        if points is not None and not reload:
            self.points = points
        else:
            if not reload:
                try:
                    # try to read previously saved pickle
                    with open(dump, 'rb') as handler:
                        self.points = pickle.load(handler)
                except:
                    reload = True

            if reload:
                self._csv = pd.read_csv(file, delimiter=delimiter, parse_dates=[timestamp], date_parser=lambda col: pd.to_datetime(col, utc=True))
                self.points = gpd.GeoDataFrame(self._csv.drop(columns=[lon, lat]), crs=crs,
                                          geometry=gpd.points_from_xy(self._csv[lon], self._csv[lat]))
                self.points = self.points.to_crs(reproject_crs)

                with open(dump, 'wb') as handler:
                    pickle.dump(self.points, handler)
        
        self.groupby = groupby
        
        if self.groupby is not None:
            self.points_group = self.points.groupby(self.groupby)
    
    def get_track(self, track_selector=None, prepare=False, prepare_type='circle', prepare_buffer=None, anonymize=False, anonymize_buffer=(300,)):
        if track_selector is None:
            track_points = self.points
        else:
            track_points = self.points_group.get_group(track_selector)

        if anonymize:
            track_points, _ = self.anonymize(track_points, anonymize_buffer=anonymize_buffer)

        if prepare:
            track_points, _ = self.prepare(track_points, prepare_type=prepare_type, prepare_buffer=prepare_buffer)
            
        track_line = self.points_to_line(track_points)
        
        return track_points, track_line
    
    def points_to_line(self, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if len(points) <= 0:
            return None

        pairs = list(zip(itertools.repeat(self.groupby), points[:-1].itertuples(), points[1:].itertuples()))
        if self.groupby is None:
            data = [{'u': a.Index, 'v': b.Index} for _, a, b in pairs]
        else:
            data = [{**{e: getattr(a, e) for e in g}, 'u': a.Index, 'v': b.Index} for g, a, b in pairs]
        line = [shp.geometry.LineString([(a.geometry.x, a.geometry.y), (b.geometry.x, b.geometry.y)]) for _, a, b in pairs]
        gdf = gpd.GeoDataFrame(data, geometry=line, crs=points.crs)
        gdf['length'] = gdf['geometry'].length
        gdf['bearing'] = self.calculate_bearing(gdf)
        return gdf

    def calculate_angle_xy(self, a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def calculate_angle_points(self, a: shp.geometry.Point, b: shp.geometry.Point) -> float:
        return self.calculate_angle_xy((a.x, a.y), (b.x, b.y))

    def calculate_bearing_points(self, a: shp.geometry.Point, b: shp.geometry.Point) -> float:
        return self.calculate_bearing_xy((a.x, a.y), (b.x, b.y))

    def calculate_bearing(self, gdf: gpd.GeoDataFrame) -> pd.Series:
        # calculate azimuth
        if len(gdf) <= 0:
            return None

        alpha = gdf['geometry'].apply(lambda line: math.atan2(
                line.coords[1][0] - line.coords[0][0],
                line.coords[1][1] - line.coords[0][1]))
        # add 360 and mod for positive bearing from 0 to 360 degrees
        return np.degrees(alpha).add(360).mod(360).rename('bearing')
    
    def create_filter_triangle(self, a: shp.geometry.Point, b: shp.geometry.Point, buffer_width: None, buffer_length: None):
        if buffer_width is None:
            buffer_width = a.distance(b)
        if buffer_length is None:
            buffer_length = a.distance(b) * 2
        
        angle_ab = self.calculate_angle_points(a, b)
        translate_1_x = math.cos(angle_ab)
        translate_1_y = math.sin(angle_ab)
        point_1 = shp.affinity.translate(a, translate_1_x * buffer_length, translate_1_y * buffer_length)
        
        angle_2 = angle_ab + math.pi / 2 # rotate 90 degrees to left
        translate_2_x = math.cos(angle_2)
        translate_2_y = math.sin(angle_2)
        point_2 = shp.affinity.translate(a, translate_2_x * buffer_width, translate_2_y * buffer_width)
        
        angle_3 = angle_ab - math.pi / 2 # rotate 90 degrees to right
        translate_3_x = math.cos(angle_3)
        translate_3_y = math.sin(angle_3)
        point_3 = shp.affinity.translate(a, translate_3_x * buffer_width, translate_3_y * buffer_width)
        
        return shp.geometry.Polygon([(point_1.x, point_1.y), (point_2.x, point_2.y), (point_3.x, point_3.y), (point_1.x, point_1.y)])
    
    def prepare(self, points: gpd.GeoDataFrame, prepare_type='circle', prepare_buffer=None, remove_timestamp=True, remove_position=True, remove_forward_backward=True, forward_backward_angle=120) -> gpd.GeoDataFrame:
        # sort by timestamp for making sure that the track follows the law of time
        points = points.sort_values('timestamp')

        sindex = points.sindex
        sindex_ids = {points.index[i]: i for i in range(len(points))}
        sindex_names = [points.index[i] for i in range(len(points))]
        sindex_ids = collections.OrderedDict(sorted(sindex_ids.items(), key=lambda t: t[0]))

        if remove_timestamp:
            # remove points with same timestamp, they could not be sorted correctly
            points_duplicated_timestamp = points[points.duplicated('timestamp')]
            for point in points_duplicated_timestamp.itertuples():
                if point.Index in sindex_ids:
                    sindex.delete(sindex_ids[point.Index], point.geometry.bounds)
                    del sindex_ids[point.Index]

        if remove_position:
            # remove duplicate points near to each other, same geometry has no new information for the track shape in this case
            points_duplicated_geometry = []
            previous_point = None
            for point in points.itertuples():
                if previous_point is not None and point.geometry == previous_point.geometry:
                    points_duplicated_geometry.append(point.Index)
                previous_point = point

            for index in points_duplicated_geometry:
                if index in sindex_ids:
                    sindex.delete(sindex_ids[index], points.at[index, 'geometry'].bounds)
                    del sindex_ids[index]
            points = points.drop(points_duplicated_geometry)

        # re-set corrected index
        points._sindex = sindex
        points._sindex_generated = True

        point_filters = None

        if prepare_type is not None:
            if prepare_buffer is None:
                pairs = list(zip(points[:-1].itertuples(), points[1:].itertuples()))
                distances = list(filter(lambda x: x > 0, [a.geometry.distance(b.geometry) for a, b in pairs]))
                # MAD (median absolute deviation), see: https://en.wikipedia.org/wiki/Median_absolute_deviation
                median = statistics.median(distances)
                mad = statistics.median([abs(x - median) for x in distances])
                standard_deviation = 1.4826 * mad
                
                # calculate triangle height from circle area
                circle_area = math.pi * math.pow(standard_deviation / 2, 2)
                triangle_height = circle_area / (0.5 * standard_deviation)
                
                # buffer is standard deviation, for triangle, use double calculated length
                prepare_buffer = (standard_deviation, 2 * triangle_height)
            
            if prepare_type == 'triangle' and len(prepare_buffer) < 2:                
                raise ValueError("prepare_type is set to {} but prepare_buffer is too short, only contains {} but needs more axis.".format(prepare_type, prepare_buffer))
            if type(prepare_buffer) is not tuple:
                prepare_buffer = (prepare_buffer,)

            safe_points = {min(sindex_ids), max(sindex_ids)}
            points_index = list(sindex_ids.keys())
            
            point_filters = []
            
            i = 0
            while i < len(points_index) - 1:
                if prepare_type == 'triangle':
                    point_a = points.at[points_index[i], 'geometry']
                    point_b = points.at[points_index[i+1], 'geometry']
                    point_filter = self.create_filter_triangle(point_a, point_b, buffer_width=prepare_buffer[0], buffer_length=prepare_buffer[1])
                elif prepare_type == 'circle':
                    point_filter = points.at[points_index[i], 'geometry'].buffer(prepare_buffer[0])
                else:
                    raise ValueError("prepare_type is set to {} but needs a value of either 'triangle' or 'circle'.".format(prepare_type))
                
                point_filters.append(point_filter)
                
                points_preselection_indices = list(points.sindex.intersection(point_filter.bounds))
                points_preselection_names = [sindex_names[i] for i in points_preselection_indices]
                points_preselection = points.loc[points_preselection_names]

                points_within = points_preselection[points_preselection.within(point_filter)].index.to_list()
                points_within = sorted(points_within)
                
                points_within_filtered = set()
                j = 0 # for points_within
                k = 0 # for break detection
                start = False # marker for finding current point in list
                while j < len(points_within):
                    while j < len(points_within) and not start:
                        while j < len(points_within) and points_within[j] <= points_index[i]:
                            j += 1
                        if j < len(points_within):
                            k = 1
                        start = True # found first point that might be good for dropping
                    if j < len(points_within) and start:
                        if i+k >= len(points_index) or points_within[j] != points_index[i+k]:
                            break # detected gap in track (or end), skip the rest as it is not from the current movement
                        if points_within[j] not in safe_points:
                            points_within_filtered.add(points_within[j]) # found safe point to remove
                            
                            # correct sindex
                            sindex.delete(sindex_ids[points_within[j]], points.at[points_within[j], 'geometry'].bounds)
                            del sindex_ids[points_within[j]]
                        k += 1
                        j += 1

                points_index = list(sindex_ids.keys())
                i += 1

            points = points.loc[points_index]
            points._sindex = sindex
            points._sindex_generated = True

        # remove forward backward movements
        if remove_forward_backward:
            points_forward_backward_geometry = []
            previous_point = None
            previous_bearing = None
            previous_angle = None
            for point in points.itertuples():
                if previous_point is not None:
                    bearing = self.calculate_bearing_points(point.geometry, previous_point.geometry)
                    if previous_bearing is not None:
                        angle = self.angle_diff(bearing, previous_bearing)
                        if abs(angle) > forward_backward_angle and (
                                previous_angle is None or abs(previous_angle) > forward_backward_angle):
                            points_forward_backward_geometry.append(previous_point.Index)
                        previous_angle = angle
                    previous_bearing = bearing
                previous_point = point

            for index in points_forward_backward_geometry:
                if index in sindex_ids:
                    sindex.delete(sindex_ids[index], points.at[index, 'geometry'].bounds)
                    del sindex_ids[index]
            points = points.drop(points_forward_backward_geometry)

            points._sindex = sindex
            points._sindex_generated = True

        return points, point_filters

    def anonymize(self, points: gpd.GeoDataFrame, anonymize_buffer=(300,)) -> gpd.GeoDataFrame:
        # sort by timestamp for making sure that the track follows the law of time
        points = points.sort_values('timestamp')

        sindex = points.sindex
        sindex_ids = {points.index[i]: i for i in range(len(points))}
        sindex_names = [points.index[i] for i in range(len(points))]
        sindex_ids = collections.OrderedDict(sorted(sindex_ids.items(), key=lambda t: t[0]))

        if type(anonymize_buffer) is not tuple:
            anonymize_buffer = (anonymize_buffer,)

        points_index = list(sindex_ids.keys())
        # first and last point removed with anonymizing circle around
        point_filters = [points.at[points_index[0], 'geometry'].buffer(anonymize_buffer[0]),
                         points.at[points_index[-1], 'geometry'].buffer(anonymize_buffer[0])]

        reverse = False
        for point_filter in point_filters:
            points_preselection_indices = list(points.sindex.intersection(point_filter.bounds))
            points_preselection_names = [sindex_names[i] for i in points_preselection_indices]
            points_preselection = points.loc[points_preselection_names]

            points_within = points_preselection[points_preselection.within(point_filter)].index.to_list()
            points_within = sorted(points_within)
            if reverse:
                points_within = list(reversed(points_within))

            points_within_filtered = set()
            i = 0 if not reverse else -1  # for points_index
            j = 0  # for points_within
            k = 0  # for break detection
            while j < len(points_within):
                if points_within[j] != points_index[i + k]:
                    break  # detected gap in track (or end), skip the rest as it is not from the current movement
                points_within_filtered.add(points_within[j])  # found point to remove

                # correct sindex
                sindex.delete(sindex_ids[points_within[j]], points.at[points_within[j], 'geometry'].bounds)
                del sindex_ids[points_within[j]]
                k += 1 if not reverse else -1
                j += 1

            reverse = True

        points_index = list(sindex_ids.keys())
        points = points.loc[points_index]
        points._sindex = sindex
        points._sindex_generated = True

        return points, point_filters

    def export_prepare(self, graph: nx.MultiDiGraph, points: gpd.GeoDataFrame, prepare_type='circle', prepare_buffer=None, plot=True, node_size=1, edge_linewidth=1, image_file="images/prepare.png", image_format="png", dpi=300, show=True, export=True, export_file_pattern="exports/prepare_{}.geojson", export_driver="GeoJSON"):
        prepared, filters = self.prepare(points, prepare_type=prepare_type, prepare_buffer=prepare_buffer)
        
        filters = gpd.GeoDataFrame(geometry=filters, crs=graph.graph['crs'])
        dropped_points = points[~points.isin(prepared)].dropna()
        
        if plot:
            points_bounds = points.total_bounds
            filters_bounds = filters.total_bounds
            # north, south, east, west
            bbox = (max(points_bounds[3], filters_bounds[3]),
                    min(points_bounds[1], filters_bounds[1]),
                    max(points_bounds[2], filters_bounds[2]),
                    min(points_bounds[0], filters_bounds[0]))
            
            filters_boundary = filters['geometry'].boundary
            
            # plot network
            fig, ax = ox.plot_graph(graph, bbox=bbox, node_size=node_size*15, edge_linewidth=edge_linewidth, node_color='#999999', margin=0.1, axis_off=False, show=False, close=False)
            
            # plot prepare
            filters_boundary.plot(ax=ax, color='orange', linewidth=edge_linewidth*0.2, zorder=4)
            dropped_points.plot(ax=ax, color='purple', markersize=node_size, edgecolors='none', zorder=5)
            prepared.plot(ax=ax, color='red', markersize=node_size, edgecolors='none', zorder=10)
            
            plt.savefig(image_file, dpi=dpi, format=image_format)
            if show:
                plt.show()
            else:
                plt.close()
        
        if export:
            filters.to_file(export_file_pattern.format('filters'), driver=export_driver)
            if not dropped_points.empty:
                dropped_points.to_file(export_file_pattern.format('dropped'), driver=export_driver)
            prepared.to_file(export_file_pattern.format('filtered'), driver=export_driver)
