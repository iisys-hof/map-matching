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

from builtins import int

import math

import geopandas as gpd
import shapely as shp
import shapely.geometry
import shapely.ops


class GeoData:

    def calculate_bearing_xy(self, a, b):
        # calculate azimuth
        alpha = math.atan2(b[0] - a[0], b[1] - a[1])

        # add 360 and mod for positive bearing from 0 to 360 degrees
        return (math.degrees(alpha) + 360) % 360

    def calculate_bearing_points(self, a: shp.geometry.Point, b: shp.geometry.Point) -> float:
        return self.calculate_bearing_xy((a.x, a.y), (b.x, b.y))

    def calculate_bearing_line(self, line: shp.geometry.LineString) -> float:
        return self.calculate_bearing_xy(line.coords[0], line.coords[-1])

    # right is negative from 0 to -179, left is positive from 0 to 180
    def angle_diff(self, a, b):
        d = a - b
        return d - 360.0 if d >= 180.0 else d + 360.0 if d <= -180.0 else d

    def substring_line(self, a: shp.geometry.LineString, b: shp.geometry.LineString):
        if len(a.coords) == 0 or len(b.coords) == 0:
            return shp.geometry.LineString(), a, shp.geometry.LineString()

        # get first and last point
        b_0 = shp.geometry.Point(b.coords[0])
        b_1 = shp.geometry.Point(b.coords[-1])

        # cut a with b
        a_start = shp.ops.substring(a, 0, a.project(b_0))
        a_sub = shp.ops.substring(a, a.project(b_0), a.project(b_1))
        a_end = shp.ops.substring(a, a.project(b_1), a.length)

        return a_start, a_sub, a_end

    def _split_lines_if_necessary(self, points: list, lines: list, split_points: gpd.GeoSeries):
        if len(split_points) == 0:
            return

        i = 0
        while i < len(lines):
            point_selection = split_points.loc[list(split_points.sindex.intersection(lines[i].buffer(1e-05).bounds))]
            for point in point_selection:
                if not shp.geometry.Point(lines[i].coords[0]).almost_equals(point) and \
                        not shp.geometry.Point(lines[i].coords[-1]).almost_equals(point):
                    position = lines[i].project(point)
                    interpolate = lines[i].interpolate(position)
                    if point.almost_equals(interpolate):
                        # only contained if within interior but not in boundary
                        line_start = shp.ops.substring(lines[i], 0, position)
                        line_end = shp.ops.substring(lines[i], position, lines[i].length)
                        if line_start.length > 0 and line_end.length > 0:
                            # only use split if it was successful
                            lines[i] = line_start
                            lines.insert(i + 1, line_end)
                            points.insert(i + 1, interpolate)
            i += 1

    def _fix_missing_points(self, a_points: list, a_lines: list, b_points: list, b_lines: list):
        # copy points for being able to use minimal set later
        a_split_points = gpd.GeoSeries(a_points)
        b_split_points = gpd.GeoSeries(b_points)
        # split lines by points of themselves for reverse and loop parts overlapping
        self._split_lines_if_necessary(a_points, a_lines, a_split_points)
        self._split_lines_if_necessary(b_points, b_lines, b_split_points)
        # split lines by points of the other line copied above for eliminating overlaps between lines
        self._split_lines_if_necessary(a_points, a_lines, b_split_points)
        self._split_lines_if_necessary(b_points, b_lines, a_split_points)

    def _detect_reverse(self, lines, start=0, end=-1):
        end = len(lines) if end == -1 else end

        r_start = -1
        r_turn = -1
        r_end = -1
        r_start_point = None

        if start + 1 < end and \
                (lines[start].contains(lines[start + 1]) or lines[start + 1].contains(lines[start])):
            # reverse detected
            if start == 0:
                r_start = 0
            else:
                r_start = start + 1
            r_start_point = shp.geometry.Point(lines[r_start].coords[0])

            for i in range(r_start, end):
                if r_turn == -1 and i + 1 < end and \
                        (lines[i].contains(lines[i + 1]) or lines[i + 1].contains(lines[i])):
                    # turn detected
                    r_turn = i + 1
                    r_turn_point = shp.geometry.Point(lines[r_turn].coords[0])
                if r_turn != -1 and lines[i].touches(r_start_point) or lines[i].contains(r_start_point):
                    # end point detected
                    if i + 1 < end and shp.geometry.Point(lines[i].coords[-1]).almost_equals(r_start_point):
                        r_end = i + 1
                    elif shp.geometry.Point(lines[i].coords[0]).almost_equals(r_start_point):
                        r_end = i

                # when we found an end, stop reverse tracking
                if r_end != -1:
                    break

            # if we did not find a turn, end might be wrong, research again without turn
            if r_turn == -1:
                for i in range(r_start, end):
                    if lines[i].touches(r_start_point) or lines[i].contains(r_start_point):
                        # end point detected
                        if i + 1 < end and shp.geometry.Point(lines[i].coords[-1]).almost_equals(r_start_point):
                            r_end = i + 1
                        elif shp.geometry.Point(lines[i].coords[0]).almost_equals(r_start_point):
                            r_end = i
            # if we did find a turn but it was right after start, then set end to turn, track continues there
            elif r_start == 0:
                r_end = r_turn

            # if we still did not find an end, use given end
            if r_end == -1:
                r_end = end

            # if the end is the same as the start within the line, we had a two line forward-back reverse
            if r_start != start and r_end != end - 1 and r_start == r_end:
                r_start = start
                r_end = r_start + 2  # end is next after

        return r_start != -1 and r_end != -1, r_start, r_turn, r_end

    def _find_merge(self, a_points, b_points, a_start=0, a_end=-1, b_start=0, b_end=-1):
        a_end = len(a_points) if a_end == -1 else a_end
        b_end = len(b_points) if b_end == -1 else b_end

        if a_points[a_start].almost_equals(b_points[b_start]) \
                and a_start + 1 < a_end and b_start + 1 < b_end \
                and not a_points[a_start + 1].almost_equals(b_points[b_start + 1]):
            b_start += 1

        for j in range(b_start, b_end):
            b_point = b_points[j]
            for i in range(a_start, a_end):
                a_point = a_points[i]
                if a_point.almost_equals(b_point):
                    return i, j

        return len(a_points) - 1, len(b_points) - 1

    def _accumulate_error(self, lines, start=0, end=-1):
        end = len(lines) if end == -1 else end
        return [lines[i] for i in range(start, end)]

    def compare_matches(self, a: shp.geometry.LineString, b: shp.geometry.LineString, sub=True):
        if len(a.coords) == 0 and len(b.coords) == 0:
            return 0.0, 0.0, 0.0
        elif len(b.coords) == 0:
            return 0.0, a.length, float('inf')

        if sub:
            _, a, _ = self.substring_line(a, b)

        # debug flag
        debug = False

        # generate points from lines
        a_points = [shp.geometry.Point(point) for point in a.coords]
        a_lines = [shp.geometry.LineString([a_points[i], a_points[i + 1]]) for i in range(len(a_points) - 1)]
        b_points = [shp.geometry.Point(point) for point in b.coords]
        b_lines = [shp.geometry.LineString([b_points[i], b_points[i + 1]]) for i in range(len(b_points) - 1)]

        error_misses = []
        error_adds = []

        reverse_a_lines = []
        reverse_b_lines = []

        if debug:
            print("Before fixing:")
            print("a: ", ", ".join([str(a_line) for a_line in a_lines]))
            print("b: ", ", ".join([str(b_line) for b_line in b_lines]))
            print("a: ", ", ".join([str(a_point) for a_point in a_points]))
            print("b: ", ", ".join([str(b_point) for b_point in b_points]))

        self._fix_missing_points(a_points, a_lines, b_points, b_lines)

        if debug:
            print("After fixing:")
            print("a: ", ", ".join([str(a_line) for a_line in a_lines]))
            print("b: ", ", ".join([str(b_line) for b_line in b_lines]))
            print("a: ", ", ".join([str(a_point) for a_point in a_points]))
            print("b: ", ", ".join([str(b_point) for b_point in b_points]))
            print("")

        i = 0
        j = 0
        while i < len(a_lines) and j < len(b_lines):
            a_reverse, a_r_s, a_r_t, a_r_e = self._detect_reverse(a_lines, i)
            b_reverse, b_r_s, b_r_t, b_r_e = self._detect_reverse(b_lines, j)

            i_new_a = i
            j_new_a = j
            i_new_b = i
            j_new_b = j
            if a_reverse:
                if debug:
                    print("reverse in a detected at", "i:", i, ", j:", j, "with:", a_r_s, a_r_t, a_r_e)
                reverse_a_lines.extend([a_lines[i] for i in range(a_r_s, a_r_e)])
                if a_points[i].almost_equals(b_points[j]) and a_points[a_r_e].almost_equals(b_points[j + 1]):
                    j_new_a += 1
                i_new_a = a_r_e
            if b_reverse:
                if debug:
                    print("reverse in b detected at", "i:", i, ", j:", j, "with:", b_r_s, b_r_t, b_r_e)
                reverse_b_lines.extend([b_lines[i] for i in range(b_r_s, b_r_e)])
                if a_points[i].almost_equals(b_points[j]) and a_points[i + 1].almost_equals(b_points[b_r_e]):
                    i_new_b += 1
                j_new_b = b_r_e
            i = max(i_new_a, i_new_b)
            j = max(j_new_a, j_new_b)

            if i >= len(a_lines) or j >= len(b_lines):
                break

            if not a_points[i].almost_equals(b_points[j]) or not a_points[i + 1].almost_equals(b_points[j + 1]):
                # b point is somewhere outside of a
                i_new, j_new = self._find_merge(a_points, b_points, a_start=i, b_start=j)
                if debug:
                    print("outside path from", "i:", i, ", j:", j, "to", "i_new:", i_new, ", j_new:", j_new)
                error_misses.extend(self._accumulate_error(a_lines, i, i_new))
                error_adds.extend(self._accumulate_error(b_lines, j, j_new))
                i = i_new
                j = j_new
            else:
                # equal parts, no error except when we were outside
                i += 1
                j += 1

        if i < len(a_lines):
            if debug:
                print("missing from", "i:", i, "to", len(a_lines))
            error_misses.extend(self._accumulate_error(a_lines, i))
        if j < len(b_lines):
            if debug:
                print("added from", "j:", j, "to", len(b_lines))
            error_adds.extend(self._accumulate_error(b_lines, j))

        if debug:
            print("Before reverse correction:")
            print("D: ", ", ".join([str(line) for line in reverse_a_lines]))
            print("R: ", ", ".join([str(line) for line in reverse_b_lines]))
            print("A: ", ", ".join([str(line) for line in error_adds]))
            print("M: ", ", ".join([str(line) for line in error_misses]))

        # reverses that are part of both lines were correct
        i = 0
        while i < len(reverse_b_lines):
            j = 0
            while j < len(reverse_a_lines):
                if reverse_b_lines[i].almost_equals(reverse_a_lines[j]):
                    del reverse_b_lines[i]
                    del reverse_a_lines[j]
                    i = -1  # start over
                    break
                j += 1
            i += 1

        # supposed added parts that were part the a reverse line were in fact correct
        i = 0
        while i < len(error_adds):
            j = 0
            while j < len(reverse_a_lines):
                if error_adds[i].almost_equals(reverse_a_lines[j]):
                    del error_adds[i]
                    del reverse_a_lines[j]
                    i = -1  # start over
                    break
                j += 1
            i += 1

        # supposed missed parts that were part the b reverse line were in fact correct
        i = 0
        while i < len(error_misses):
            j = 0
            while j < len(reverse_b_lines):
                if error_misses[i].almost_equals(reverse_b_lines[j]):
                    del error_misses[i]
                    del reverse_b_lines[j]
                    i = -1  # start over
                    break
                j += 1
            i += 1

        # remaining duplicate lines are missed lines because already matched duplicate lines were removed above
        error_misses.extend([a_line for a_line in reverse_a_lines if a_line not in error_misses])

        # remaining reverse lines were added erroneously
        error_adds.extend([b_line for b_line in reverse_b_lines if b_line not in error_adds])

        if debug:
            print("After reverse correction:")
            print("D: ", ", ".join([str(line) for line in reverse_a_lines]))
            print("R: ", ", ".join([str(line) for line in reverse_b_lines]))
            print("A: ", ", ".join([str(line) for line in error_adds]))
            print("M: ", ", ".join([str(line) for line in error_misses]))

        error_add = sum([line.length for line in error_adds])
        error_miss = sum([line.length for line in error_misses])

        error_fraction = (error_add + error_miss) / a.length if a.length > 0 else float('inf')

        return error_add, error_miss, error_fraction
