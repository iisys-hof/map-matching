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
import bisect

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely as shp
import matplotlib.pyplot as plt

import lib.geodata

class Network(lib.geodata.GeoData):
    
    def __init__(self, name, result=1, type="drive", file=os.path.join("data", "network.graphml"), graph=None,
                 reload=False, simplify=False, strict=True, retain_all=True):
        super().__init__()
        
        if graph is not None and not reload:
            self.graph = graph
        else:
            if not reload:
                try:
                    # try to read previously saved graph
                    self.graph = ox.load_graphml(file)
                except:
                    reload = True
            
            if reload:
                self.graph = ox.graph_from_place(name, which_result=result, network_type=type, simplify=simplify if simplify and strict else False, retain_all=retain_all)
                if simplify and not strict:
                    self.graph = ox.simplify_graph(self.graph, strict)
                ox.save_graphml(self.graph, filepath=file)
        
            # add edge lengths and bearings
            self.graph = ox.utils_graph.add_edge_lengths(self.graph)
            self.graph = ox.bearing.add_edge_bearings(self.graph)
            # reproject data to UTM
            self.graph = ox.projection.project_graph(self.graph)

    def truncate_graph(self, graph: nx.MultiDiGraph, track: gpd.GeoDataFrame, buffer=1000, retain_all=False) -> nx.MultiDiGraph:
        track_bounds = track.total_bounds
        return ox.truncate.truncate_graph_bbox(graph, track_bounds[3] + buffer, track_bounds[1] - buffer, track_bounds[2] + buffer, track_bounds[0] - buffer, truncate_by_edge=True, retain_all=retain_all)
    
    def _candidate_update(self, graph: nx.MultiDiGraph, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, candidates_current: list, candidates_selector=None, use_existing=True, use_new=True) -> nx.MultiDiGraph:
        if not use_existing and not use_new:
            raise ValueError("Either existing or new or both types of nodes have to be used for candidate search, but none were allowed.")
        
        if use_new:
            nodes_index = nodes['curr_index'] if 'curr_index' in nodes else max(nodes) + 1
            edges_index = edges['names']['curr_index'] if 'curr_index' in edges['names'] else max(edges['names']) + 1
        
            # index for faster update of candidates
            candidates_lines_dict = {}
            for i in range(len(candidates_current)):
                candidates_entry = candidates_current[i]
                lines = candidates_entry['lines']
                if '_lines_dict' not in candidates_entry:
                    candidates_entry['_lines_dict'] = {}
                    for name in lines:
                        if name not in candidates_lines_dict:
                            candidates_lines_dict[name] = []
                        candidates_lines_dict[name].append(i)
                        candidates_entry['_lines_dict'][(lines[name]['u'], lines[name]['v'], lines[name]['key'])] = name
        
        if candidates_selector is not None:
            candidates_selectors = [candidates_selector]
        else:
            candidates_selectors = list(range(len(candidates_current)))
        
        for i in candidates_selectors:
            candidates_entry = candidates_current[i]
            point = candidates_entry['point']
            lines = candidates_entry['lines']
            candidates = candidates_entry['candidates']
            
            remove_names = set()
            if use_new:
                lines_to_replace = []
            
            for name in candidates:
                u = lines[name]['u']
                v = lines[name]['v']
                key = lines[name]['key']
                geometry = lines[name]['geometry']

                new_node = False
                line_split = None

                split_line = shp.geometry.LineString([point['geometry'], candidates[name]['geometry']])
                fact = (split_line.length + 1.0) / split_line.length
                split_line = shp.affinity.scale(split_line, xfact=fact, yfact=fact)
                
                if use_new:
                    line_attributes = edges['names'][edges['nodes'][(u, v, key)]]
        
                    try:
                        line_split = shp.ops.split(geometry, split_line)
                    except:
                        pass

                    if line_split is not None and len(line_split) == 2:
                        # only use split if we have not split near an existing node, normal difference is between 1e-8 and 1e-11
                        if line_split[0].length > 1e-6 and line_split[1].length > 1e-6:
                            angle = self.angle_diff(self.calculate_bearing_line(split_line),
                                                    self.calculate_bearing_xy(line_split[0].coords[-2], line_split[1].coords[1]))
            
                            t = nodes_index
                            nodes_index += 1
                            
                            u_t_line_attributes = line_attributes.copy()
                            del u_t_line_attributes['u'], u_t_line_attributes['v'], u_t_line_attributes['key']
                            u_t_line_attributes['geometry'] = line_split[0]
                            u_t_line_attributes['length'] = line_split[0].length
                            u_t_line_attributes['bearing'] = self.calculate_bearing_line(line_split[0])
                            
                            t_v_line_attributes = line_attributes.copy()
                            del t_v_line_attributes['u'], t_v_line_attributes['v'], t_v_line_attributes['key']
                            t_v_line_attributes['geometry'] = line_split[1]
                            t_v_line_attributes['length'] = line_split[1].length
                            t_v_line_attributes['bearing'] = self.calculate_bearing_line(line_split[1])
            
                            # correct graph
                            graph.add_node(t, x=candidates[name]['geometry'].x, y=candidates[name]['geometry'].y)
                            graph.add_edge(u, t, key, **u_t_line_attributes)
                            graph.add_edge(t, v, key, **t_v_line_attributes)
                            graph.remove_edge(u, v, key)
                            
                            # correct nodes
                            t_node = {'x': candidates[name]['geometry'].x, 'y': candidates[name]['geometry'].y, 'geometry': candidates[name]['geometry']}
                            nodes[t] = t_node
                            
                            # correct edges
                            u_t_edge = {'u': u, 'v': t, 'key': key, **u_t_line_attributes}
                            t_v_edge = {'u': t, 'v': v, 'key': key, **t_v_line_attributes}
                            edges['names'][edges_index] = u_t_edge
                            edges['names'][edges_index + 1] = t_v_edge
                            if (u, t) not in edges['keys']:
                                edges['keys'][(u, t)] = []
                            bisect.insort(edges['keys'][(u, t)], key)
                            if (t, v) not in edges['keys']:
                                edges['keys'][(t, v)] = []
                            bisect.insort(edges['keys'][(t, v)], key)
                            edges['nodes'][(u, t, key)] = edges_index
                            edges['nodes'][(t, v, key)] = edges_index + 1
                            edges_index += 2
                            lines_to_replace.append((name, (u, v, key), u_t_edge, t_v_edge))
        
                            candidates[name]['node'] = t
                            candidates[name]['angle'] = angle
                            new_node = True
    
                if not new_node:
                    if use_existing:
                        # if we could not introduce a new node, use the nearest next
                        if candidates[name]['projection'] < lines[name]['length'] / 2:
                            candidates[name]['node'] = u
                        else:
                            candidates[name]['node'] = v
                        
                        angle = self.angle_diff(self.calculate_bearing_line(split_line),
                                                self.calculate_bearing_line(geometry))
                        candidates[name]['angle'] = angle
                    else:
                        # we did not find a new node and are not allowed to use existing, delete edge
                        remove_names.add(name)
            
            # search edges with same node, they are duplicates and don't add new information for routing
            duplicate_nodes = set()
            for name in candidates_entry['candidates']:
                if name not in remove_names:
                    duplicate_node = candidates_entry['candidates'][name]['node']
                    if duplicate_node in duplicate_nodes:
                        remove_names.add(name)
                    duplicate_nodes.add(duplicate_node)
            
            # remove duplicates
            for name in remove_names:
                line = candidates_entry['lines'][name]
                selector = (line['u'], line['v'], line['key'])

                del candidates_entry['candidates'][name]
                if '_lines_dict' in candidates_entry:
                    del candidates_entry['_lines_dict'][selector]
                del candidates_entry['lines'][name]
            
            candidates_entry['nodes'] = {candidates[name]['node']: nodes[candidates[name]['node']] for name in candidates}
            candidates_entry['names'] = [name for name in candidates]
            
            if use_new:
                # correct candidates_list with updates
                for name, selector, u_t_edge, t_v_edge in lines_to_replace:
                    line_candidates_selectors = candidates_lines_dict[name]
                    for i in line_candidates_selectors:
                        candidates_entry = candidates_current[i]
                        if selector in candidates_entry['_lines_dict']:
                            name = candidates_entry['_lines_dict'][selector]
                            if candidates_entry['candidates'][name]['projection'] <= u_t_edge['length']:
                                candidates_entry['lines'][name].update(u_t_edge)
                                candidates_entry['_lines_dict'][(u_t_edge['u'], u_t_edge['v'], u_t_edge['key'])] = name
                                candidates_entry['candidates'][name]['projection'] = u_t_edge['geometry'].project(candidates_entry['point']['geometry'])
                            else:
                                candidates_entry['lines'][name].update(t_v_edge)
                                candidates_entry['_lines_dict'][(t_v_edge['u'], t_v_edge['v'], t_v_edge['key'])] = name
                                candidates_entry['candidates'][name]['projection'] = t_v_edge['geometry'].project(candidates_entry['point']['geometry'])
                            del candidates_entry['_lines_dict'][selector]

        # remove empty candidates, if that happened
        remove_candidates = []
        for i in candidates_selectors:
            candidates_entry = candidates_current[i]
            length = len(candidates_entry['names'])
            if length == 0:
                remove_candidates.append(candidates_entry)
        for candidates_entry in remove_candidates:
            candidates_current.remove(candidates_entry)

        if use_new:
            nodes['curr_index'] = nodes_index
            edges['names']['curr_index'] = edges_index

        return graph, nodes, edges, candidates_current
    
    def candidate_search(self, graph_clean: nx.MultiDiGraph, graph_updated=None, nodes_clean=None, nodes_updated=None, edges_clean=None, edges_updated=None, points=None, candidates_selections=None, candidates_current=None, buffer=50, buffer_limit=10000, use_existing=True, use_new=True):
        if points is None and candidates_selections is None:
            raise ValueError("Either points or candidates_selections has to be set, but none were.")
        
        if candidates_selections is not None and (candidates_current is None or graph_updated is None):
            raise ValueError("When a list of candidates shall be re-searched, candidates_current and graph_updated have to be provided, too.")
        
        if graph_updated is not None and (nodes_updated is None or edges_updated is None):
            raise ValueError("When graph_updated is set, nodes_updated and edges_updated have to be given, too, because they have a unique index that cannot be rebuilt.")

        # extract edges for candidate search
        if nodes_clean is None and edges_clean is None:
            nodes_clean, edges_clean = ox.graph_to_gdfs(graph_clean)
        elif nodes_clean is None:
            nodes_clean = ox.graph_to_gdfs(graph_clean, nodes=True, edges=False)
        elif edges_clean is None:
            edges_clean = ox.graph_to_gdfs(graph_clean, nodes=False, edges=True)
        
        if candidates_selections is not None and type(candidates_selections) is not list:
            candidates_selections = [candidates_selections]
        
        # if we search for several points, select them from candidates
        if candidates_selections is not None:
            points = [candidates_current[candidates_selection]['point'] for candidates_selection in candidates_selections]
        
        if type(points) is gpd.GeoDataFrame:
            points = points.to_dict(orient='index')

        # search for points (or single point if set before)
        candidates_list = [] if candidates_current is None else candidates_current
        i = 0
        for name in points:
            point = points[name]
            lines = None
            buffer_current = buffer

            # repeat until we found lines or reached buffer_limit
            while lines is None or len(lines) == 0 and buffer_current * 2 <= buffer_limit:
                # calculate buffer around the point for selection
                point_buffer = point['geometry'].buffer(buffer_current)

                # use index structure of lines to preselect the lines within the buffer bounds
                edges_preselection_indices = list(edges_clean.sindex.intersection(point_buffer.bounds))
                edges_preselection = edges_clean.loc[edges_preselection_indices]
                
                # query concrete lines intersecting the buffer
                lines = edges_preselection[edges_preselection.intersects(point_buffer)]

                buffer_current *= 2

            if len(lines) > 0:
                # project point onto lines and calculate points on the lines plus distance
                line_projection = lines.project(point['geometry'])
                line_points = lines.interpolate(line_projection)
                line_distances = line_points.distance(point['geometry'])
        
                line_distances_sorted = line_distances.sort_values()
                
                candidates = collections.OrderedDict()
                for name, distance in line_distances_sorted.items():
                    candidates[name] = {'distance': distance, 'projection': line_projection.at[name], 'geometry': line_points.at[name]}

                candidates_entry = {'point': point, 'lines': lines.to_dict(orient='index'), 'candidates': candidates}
                
                if candidates_current is None:
                    candidates_list.append(candidates_entry)
                else:
                    candidates_list[candidates_selections[i]] = candidates_entry
            i += 1

        # create updated copies if not set
        if graph_updated is None:
            if use_new:
                graph_updated = graph_clean.copy()
            else:
                graph_updated = graph_clean
            nodes_updated = nodes_clean.to_dict(orient='index')
            edges_updated = self._edges_to_dict(edges_clean)

        graph_updated, nodes_updated, edges_updated, candidates_list = self._candidate_update(graph_updated, nodes_updated, edges_updated, candidates_current=candidates_list, use_existing=use_existing, use_new=use_new)

        return nodes_clean, edges_clean, candidates_list, graph_updated, nodes_updated, edges_updated

    def _edges_to_dict(self, edges: gpd.GeoDataFrame) -> dict:
        edges_name = edges.to_dict(orient='index')

        edges_keys = {}
        for name in edges_name:
            edges_key = (edges_name[name]['u'], edges_name[name]['v'])
            if edges_key not in edges_keys:
                edges_keys[edges_key] = []
            bisect.insort(edges_keys[edges_key], edges_name[name]['key'])

        edges_nodes = {(edges_name[name]['u'], edges_name[name]['v'], edges_name[name]['key']): name for name in edges_name}

        return {'names': edges_name, 'keys': edges_keys, 'nodes': edges_nodes}

    def route_edges_data(self, edges: dict, route: list):
        if len(route) < 2:
            return None

        route_edges_uv = list(zip(route[:-1], route[1:]))
        route_edges_names = [(u, v, edges['keys'][(u, v)][0]) for u, v in route_edges_uv]
        return [edges['names'][edges['nodes'][selector]] for selector in route_edges_names]
    
    def route(self, graph: nx.MultiDiGraph, source_candidate: dict, target_candidate: dict, source_select=0, target_select=0, weight='length', method='dijkstra', routes_cache={}) -> tuple:
        source_nodes_length = len(source_candidate['names'])
        if source_nodes_length == 0 or source_select >= source_nodes_length:
            return {'route': None, 'source_candidate': source_candidate, 'source_select': source_select, 'target_candidate': target_candidate, 'target_select': target_select}, routes_cache
        source_node = source_candidate['candidates'][source_candidate['names'][source_select]]['node']
        
        target_nodes_length = len(target_candidate['names'])
        if target_nodes_length == 0 or target_select >= target_nodes_length:
            return {'route': None, 'source_candidate': source_candidate, 'source_select': source_select, 'target_candidate': target_candidate, 'target_select': target_select}, routes_cache
        target_node = target_candidate['candidates'][target_candidate['names'][target_select]]['node']
        
        try:
            route = routes_cache[(source_node, target_node)]['route']
        except KeyError:
            try:
                route = nx.shortest_path(graph, source_node, target_node, weight=weight, method=method)
            except nx.NetworkXNoPath:
                route = None
            routes_cache[(source_node, target_node)] = {'source': source_node, 'target': target_node, 'route': route}
        
        return {'route': route, 'source_candidate': source_candidate, 'source_select': source_select, 'target_candidate': target_candidate, 'target_select': target_select}, routes_cache

    def routes_join(self, routes: list) -> list:
        # routes must be joinable, last node of previous route is first node of following route
        route = []
        for i in range(len(routes)):
            if routes[i]['route'] is None:
                raise ValueError("Route at index {} is None, cannot concat.".format(i))
            if i > 0 and routes[i]['route'][0] != routes[i-1]['route'][-1]:
                raise ValueError("Route at index {} starts at node {} but previous route ends at {}, cannot concat.".format(i, routes[i]['route'][0], routes[i-1]['route'][-1]))
            route.extend(routes[i]['route'][:-1])
            if i + 1 >= len(routes):
                route.append(routes[i]['route'][-1])
        return route

    def calculate_bearing(self, gdf: gpd.GeoDataFrame) -> pd.Series:
        return self.calculate_bearing_points(gdf['geometry'].iloc[0], gdf['geometry'].iloc[1])

    def calculate_direction(self, edges: dict, u: int, v: int, w: int, directions_cache={}) -> tuple:
        try:
            direction = directions_cache[(u, v, w)]
        except KeyError:
            n = edges['names'][edges['nodes'][(u, v, edges['keys'][(u, v)][0])]]
            m = edges['names'][edges['nodes'][(v, w, edges['keys'][(v, w)][0])]]
    
            direction = self.angle_diff(n['bearing'], m['bearing'])
            directions_cache[(u, v, w)] = direction
        
        return direction, directions_cache
    
    def route_metrics(self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, route: list, route_metrics_cache={}) -> tuple:
        if route is None:
            return {'length': 0, 'bearing': 0, 'bearing_mean': 0}, route_metrics_cache
        
        route_index = tuple(route)
        
        try:
            route_metrics = route_metrics_cache[route_index]
        except KeyError:
            route_edges_data = self.route_edges_data(edges, route)
            route_length = sum([data['length'] for data in route_edges_data]) if route_edges_data is not None else 0
            route_bearing_mean = sum([data['bearing'] for data in route_edges_data]) / len(route_edges_data) if route_edges_data is not None else 0
            route_bearing = self.calculate_bearing_points(nodes[route[0]]['geometry'], nodes[route[-1]]['geometry']) if len(route) > 1 else 0
            route_metrics = {'length': route_length, 'bearing': route_bearing, 'bearing_mean': route_bearing_mean}
            route_metrics_cache[route_index] = route_metrics
            
        return route_metrics, route_metrics_cache

    def route_extract(self, graph: nx.MultiDiGraph, candidates: list, candidates_selection: collections.OrderedDict, routes_cache={}) -> list:
        selections = list(candidates_selection.keys())
        routes = [self.route(graph, candidates[selections[i]], candidates[selections[i+1]], candidates_selection[selections[i]], candidates_selection[selections[i+1]], routes_cache=routes_cache)[0] for i in range(len(selections) - 1)]
        for i in range(len(routes)):
            if routes[i]['route'] is None:
                routes = routes[:i]
                break
        return self.routes_join(routes)
    
    def route_line(self, graph: nx.MultiDiGraph, route: list) -> gpd.GeoDataFrame:
        lines_list = ox.plot._node_list_to_coordinate_lines(graph, route)
        lines = []
        if len(lines_list) > 0:
            for line in lines_list[:-1]:
                lines.extend(line[:-1])
            lines.extend(lines_list[-1])
        return gpd.GeoDataFrame(geometry=[shp.geometry.LineString(lines)], crs=graph.graph['crs'])

    def track_extract(self, graph: nx.MultiDiGraph, candidates: list) -> gpd.GeoDataFrame:
        points = [candidate['point'] for candidate in candidates]
        point_pairs = list(zip([point['geometry'] for point in points[:-1]], [point['geometry'] for point in points[1:]]))
        line = [shp.geometry.LineString([a, b]) for a, b in point_pairs]
        return gpd.GeoDataFrame(geometry=line, crs=graph.graph['crs'])

    def extract_nodes(self, graph: nx.MultiDiGraph, nodes: list):
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy([graph.nodes[n]['x'] for n in nodes], [graph.nodes[n]['y'] for n in nodes]), crs=graph.graph['crs'])

    def extract_edges(self, graph: nx.MultiDiGraph, edges: list):
        return gpd.GeoDataFrame([{'u': u, 'v': v, 'key': key, **graph[u][v][key]} for u, v in edges for key in range(len(graph[u][v]))], crs=graph.graph['crs'])

    def export_candidates(self, graph: nx.MultiDiGraph, candidates: list, plot=True, type='nodes', node_size=1, edge_linewidth=1, image_file="images/candidates.png", image_format="png", dpi=300, show=True, export=True, export_file_pattern="exports/candidates_{}.geojson", export_driver="GeoJSON"):
        if type == 'edges':
            point_candidate_edges = gpd.GeoDataFrame(geometry=[shp.geometry.LineString([candidate['point']['geometry'], candidate['candidates'][name]['geometry']]) for candidate in candidates for name in candidate['candidates']], crs=graph.graph['crs'])
            candidate_candidates = gpd.GeoDataFrame(geometry=[candidate['candidates'][name]['geometry'] for candidate in candidates for name in candidate['candidates']], crs=graph.graph['crs'])
        elif type == 'nodes':
            point_candidate_edges = gpd.GeoDataFrame(geometry=[shp.geometry.LineString([candidate['point']['geometry'], candidate['nodes'][candidate['candidates'][name]['node']]['geometry']]) for candidate in candidates for name in candidate['candidates']], crs=graph.graph['crs'])
            candidate_candidates = gpd.GeoDataFrame(geometry=[candidate['nodes'][candidate['candidates'][name]['node']]['geometry'] for candidate in candidates for name in candidate['candidates']], crs=graph.graph['crs'])
        else:
            raise ValueError("Parameter 'type' needs to be either 'edges' or 'nodes', but was '{}'.".format(type))
        candidate_lines = gpd.GeoDataFrame(geometry=[candidate['lines'][name]['geometry'] for candidate in candidates for name in candidate['lines']], crs=graph.graph['crs'])
        candidate_points = gpd.GeoDataFrame(geometry=[candidate['point']['geometry'] for candidate in candidates], crs=graph.graph['crs'])

        if plot:
            point_candidate_edges_bounds = point_candidate_edges.total_bounds
            candidate_candidates_bounds = candidate_candidates.total_bounds
            candidate_lines_bounds = candidate_lines.total_bounds
            candidate_points_bounds = candidate_points.total_bounds
            # north, south, east, west
            bbox = (max(point_candidate_edges_bounds[3], candidate_lines_bounds[3], candidate_candidates_bounds[3], candidate_points_bounds[3]),
                    min(point_candidate_edges_bounds[1], candidate_lines_bounds[1], candidate_candidates_bounds[1], candidate_points_bounds[1]),
                    max(point_candidate_edges_bounds[2], candidate_lines_bounds[2], candidate_candidates_bounds[2], candidate_points_bounds[2]),
                    min(point_candidate_edges_bounds[0], candidate_lines_bounds[0], candidate_candidates_bounds[0], candidate_points_bounds[0]))

            # plot network
            fig, ax = ox.plot_graph(graph, bbox=bbox, node_size=node_size*0.5, edge_linewidth=edge_linewidth*0.1, node_color='#999999', margin=0.1, axis_off=False, show=False, close=False)
    
            # plot candidates
            candidate_lines.plot(ax=ax, linewidth=edge_linewidth*0.5, zorder=2)
            candidate_candidates.plot(ax=ax, color='green', markersize=node_size, edgecolors='none', zorder=5)
            point_candidate_edges.plot(ax=ax, color='orange', linewidth=edge_linewidth*0.2, zorder=4)
            candidate_points.plot(ax=ax, color='red', markersize=node_size, edgecolors='none', zorder=10)
            
            plt.savefig(image_file, dpi=dpi, format=image_format)
            if show:
                plt.show()
            else:
                plt.close()
        
        if export:
            point_candidate_edges.to_file(export_file_pattern.format('lines'), driver=export_driver)
            candidate_candidates.to_file(export_file_pattern.format('nodes'), driver=export_driver)
            candidate_lines.to_file(export_file_pattern.format('edges'), driver=export_driver)
            candidate_points.to_file(export_file_pattern.format('points'), driver=export_driver)

    def export_track(self, graph: nx.MultiDiGraph, candidates: list, track: gpd.GeoDataFrame=None, plot=True, node_size=1, edge_linewidth=1, image_file="images/track.png", image_format="png", dpi=300, show=True, export=True, export_file_pattern="exports/track_{}.geojson", export_driver="GeoJSON"):
        # matching track
        matching_track = self.track_extract(graph, candidates)

        if plot:
            matching_track_bounds = matching_track.total_bounds
            track_bounds = track.total_bounds if track is not None else matching_track_bounds
            # north, south, east, west
            bbox = (max(track_bounds[3], matching_track_bounds[3]),
                    min(track_bounds[1], matching_track_bounds[1]),
                    max(track_bounds[2], matching_track_bounds[2]),
                    min(track_bounds[0], matching_track_bounds[0]))
            
            # plot network
            fig, ax = ox.plot_graph(graph, bbox=bbox, node_size=node_size*15, edge_linewidth=edge_linewidth, node_color='#999999', margin=0.1, axis_off=False, show=False, close=False)
    
            # plot track
            if track is not None:
                track.plot(ax=ax, color='violet', linewidth=edge_linewidth*2, alpha=0.5, zorder=2)
    
            # plot matching_track
            matching_track.plot(ax=ax, color='blue', linewidth=edge_linewidth*2, alpha=0.5, zorder=3)
            
            # plot start and end point
            candidate_extent_points = gpd.GeoDataFrame([candidates[0]['point'], candidates[-1]['point']], crs=graph.graph['crs'])
            candidate_extent_points.plot(ax=ax, color='blue', alpha=0.5, markersize=node_size*15, edgecolors='none', zorder=4)
    
            plt.savefig(image_file, dpi=dpi, format=image_format)
            if show:
                plt.show()
            else:
                plt.close()
        
        if export:
            if track is not None:
                track.to_file(export_file_pattern.format('original'), driver=export_driver)
            matching_track.to_file(export_file_pattern.format('matching'), driver=export_driver)

    def export_route(self, graph: nx.MultiDiGraph, candidates: list, candidates_selection: collections.OrderedDict, routes_cache={}, plot=True, node_size=1, edge_linewidth=1, image_file="images/route.png", image_format="png", dpi=300, show=True, export=True, export_file_pattern="exports/route_{}.geojson", export_driver="GeoJSON"):
        # route
        route = self.route_extract(graph, candidates, candidates_selection, routes_cache=routes_cache)

        if plot:
            # image bounds
            route_nodes_bounds = self.extract_nodes(graph, route).total_bounds
            # north, south, east, west
            bbox = (route_nodes_bounds[3], route_nodes_bounds[1], route_nodes_bounds[2], route_nodes_bounds[0])
            
            # plot route
            fig, ax = ox.plot_graph_route(graph, route, bbox=bbox, node_size=node_size*15, edge_linewidth=edge_linewidth, orig_dest_node_size=node_size*100, route_linewidth=edge_linewidth*4, margin=0.1, axis_off=False, show=False, close=False)
            
            plt.savefig(image_file, dpi=dpi, format=image_format)
            if show:
                plt.show()
            else:
                plt.close()
        
        if export:
            self.route_line(graph, route).to_file(export_file_pattern.format('line'), driver=export_driver)

    def export_track_route(self, graph: nx.MultiDiGraph, candidates: list, candidates_selection: collections.OrderedDict, track: gpd.GeoDataFrame=None, routes_cache={}, type='nodes', plot=True, node_size=1, edge_linewidth=1, image_file="images/track_route.png", image_format="png", dpi=300, show=True, export=True, export_file_pattern="exports/track_route_{}.geojson", export_driver="GeoJSON"):
        selections = list(candidates_selection.keys())
        if type == 'edges':
            point_candidate_edges = gpd.GeoDataFrame(geometry=[shp.geometry.LineString([candidates[selections[i]]['point']['geometry'], candidates[selections[i]]['candidates'][candidates[selections[i]]['names'][candidates_selection[selections[i]]]]['geometry']]) for i in range(len(selections))], crs=graph.graph['crs'])
            candidate_candidates = gpd.GeoDataFrame(geometry=[candidates[selections[i]]['candidates'][candidates[selections[i]]['names'][candidates_selection[selections[i]]]]['geometry'] for i in range(len(selections))], crs=graph.graph['crs'])
        elif type == 'nodes':
            point_candidate_edges = gpd.GeoDataFrame(geometry=[shp.geometry.LineString([candidates[selections[i]]['point']['geometry'], candidates[selections[i]]['nodes'][candidates[selections[i]]['candidates'][candidates[selections[i]]['names'][candidates_selection[selections[i]]]]['node']]['geometry']]) for i in range(len(selections))], crs=graph.graph['crs'])
            candidate_candidates = gpd.GeoDataFrame(geometry=[candidates[selections[i]]['nodes'][candidates[selections[i]]['candidates'][candidates[selections[i]]['names'][candidates_selection[selections[i]]]]['node']]['geometry'] for i in range(len(selections))], crs=graph.graph['crs'])
        else:
            raise ValueError("Parameter 'type' needs to be either 'edges' or 'nodes', but was '{}'.".format(type))
        candidate_points = gpd.GeoDataFrame(geometry=[candidates[i]['point']['geometry'] for i in range(len(candidates))], crs=graph.graph['crs'])

        # matching track
        matching_track = self.track_extract(graph, candidates)
        
        # route
        route = self.route_extract(graph, candidates, candidates_selection, routes_cache=routes_cache)

        if plot:
            # image bounds
            route_nodes_bounds = self.extract_nodes(graph, route).total_bounds
            matching_track_bounds = matching_track.total_bounds
            track_bounds = track.total_bounds if track is not None else matching_track_bounds
    
            # north, south, east, west
            bbox = (max(track_bounds[3], matching_track_bounds[3], route_nodes_bounds[3]),
                    min(track_bounds[1], matching_track_bounds[1], route_nodes_bounds[1]),
                    max(track_bounds[2], matching_track_bounds[2], route_nodes_bounds[2]),
                    min(track_bounds[0], matching_track_bounds[0], route_nodes_bounds[0]))
            
            # plot route
            fig, ax = ox.plot_graph_route(graph, route, bbox=bbox, node_size=node_size*15, edge_linewidth=edge_linewidth, orig_dest_node_size=node_size*100, route_linewidth=edge_linewidth*4, margin=0.1, axis_off=False, show=False, close=False)
            
            # plot track
            if track is not None:
                track.plot(ax=ax, color='violet', linewidth=edge_linewidth*2, alpha=0.5, zorder=2)
            
            # plot matching_track
            matching_track.plot(ax=ax, color='blue', linewidth=edge_linewidth*2, alpha=0.5, zorder=7)
    
            # plot start and end point
            candidate_extent_points = gpd.GeoDataFrame([candidates[0]['point'], candidates[-1]['point']], crs=graph.graph['crs'])
            candidate_extent_points.plot(ax=ax, color='blue', alpha=0.5, markersize=node_size*15, edgecolors='none', zorder=4)
        
            # plot candidates for route
            candidate_candidates.plot(ax=ax, color='green', markersize=node_size, edgecolors='none', zorder=5)
            point_candidate_edges.plot(ax=ax, color='orange', linewidth=edge_linewidth*0.2, zorder=3)
            candidate_points.plot(ax=ax, color='red', markersize=node_size, edgecolors='none', zorder=10)
    
            plt.savefig(image_file, dpi=dpi, format=image_format)
            if show:
                plt.show()
            else:
                plt.close()
        
        if export:
            if track is not None:
                track.to_file(export_file_pattern.format('original'), driver=export_driver)
            matching_track.to_file(export_file_pattern.format('matching'), driver=export_driver)
            self.route_line(graph, route).to_file(export_file_pattern.format('route_line'), driver=export_driver)
            point_candidate_edges.to_file(export_file_pattern.format('candidate_lines'), driver=export_driver)
            candidate_candidates.to_file(export_file_pattern.format('candidate_nodes'), driver=export_driver)
            candidate_points.to_file(export_file_pattern.format('points'), driver=export_driver)
