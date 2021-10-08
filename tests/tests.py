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

import unittest

import shapely as shp
import matplotlib.pyplot as plt
import geopandas as gpd
import lib.geodata


class Tests(unittest.TestCase):
    def setUp(self):
        self.geodata = lib.geodata.GeoData()

    def test_compare_matches_correct_vertical(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_correct_horizontal(self):
        a = shp.geometry.LineString([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
        b = shp.geometry.LineString([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_correct_shorted_a(self):
        a = shp.geometry.LineString([(0, 0), (3, 0), (5, 0)])
        b = shp.geometry.LineString([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_correct_shorted_b(self):
        a = shp.geometry.LineString([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
        b = shp.geometry.LineString([(0, 0), (2, 0), (5, 0)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_correct_shorted_both(self):
        a = shp.geometry.LineString([(0, 0), (2, 0), (5, 0)])
        b = shp.geometry.LineString([(0, 0), (3, 0), (5, 0)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_opposite_direction(self):
        a = shp.geometry.LineString([(0, 0), (0, 3), (0, 5)])
        b = shp.geometry.LineString([(0, 5), (0, 2), (0, 0)])
        self.assertEqual((5.0, 5.0, 10.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_missing_start(self):
        a = shp.geometry.LineString([(0, 0), (0, 2), (0, 5)])
        b = shp.geometry.LineString([(0, 1), (0, 3), (0, 5)])
        self.assertEqual((0.0, 1.0, 1.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_missing_end(self):
        a = shp.geometry.LineString([(0, 0), (0, 2), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 4)])
        self.assertEqual((0.0, 1.0, 1.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_added_start(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 5)])
        b = shp.geometry.LineString([(1, 0), (0, 0), (0, 1), (0, 2), (0, 5)])
        self.assertEqual((1.0, 0.0, 1.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_added_end(self):
        a = shp.geometry.LineString([(0, 0), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 4), (0, 5), (1, 5)])
        self.assertEqual((1.0, 0.0, 1.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_outside_added_missing(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((3.0, 1.0, 4.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_outside_part(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(1, 0), (1, 1), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4)])
        self.assertEqual((5.0, 4.0, 9.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_within(self):
        a = shp.geometry.LineString([(0, 0), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 2), (0, 1), (0, 5)])
        self.assertEqual((2.0, 0.0, 2.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_within_overlap(self):
        a = shp.geometry.LineString([(0, 0), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 2), (0, 4), (0, 2), (0, 1), (0, 5)])
        self.assertEqual((6.0, 0.0, 6.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_within_long(self):
        a = shp.geometry.LineString([(0, 0), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 4), (0, 1), (0, 5)])
        self.assertEqual((6.0, 0.0, 6.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_missing(self):
        a = shp.geometry.LineString([(0, 0), (0, 2), (0, 1), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 5)])
        self.assertEqual((0.0, 2.0, 2.0 / 7), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_missing_within(self):
        a = shp.geometry.LineString([(0, 0), (0, 2), (0, 1), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 3), (0, 2), (0, 5)])
        self.assertEqual((2.0, 2.0, 4.0 / 7), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_correct(self):
        a = shp.geometry.LineString([(0, 0), (0, 5), (0, 3), (0, 6)])
        b = shp.geometry.LineString([(0, 0), (0, 4), (0, 5), (0, 3), (0, 4), (0, 6)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_outside_correct(self):
        a = shp.geometry.LineString([(0, 0), (0, 5), (1, 5), (1, 4), (0, 4), (0, 2), (1, 2), (1, 0)])
        b = shp.geometry.LineString([(0, 0), (0, 5), (1, 5), (1, 4), (0, 4), (0, 3), (0, 2), (1, 2), (1, 0)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_outside_added(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((2.0, 0.0, 2.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_added_overlap(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (2, 1), (0, 1), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (3, 1), (0, 1), (0, 5)])
        self.assertEqual((2.0, 0.0, 2.0 / 9), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_start(self):
        a = shp.geometry.LineString([(0, 0), (0, 5)])
        b = shp.geometry.LineString([(0, 1), (0, 0), (0, 5)])
        self.assertEqual((1.0, 0.0, 1.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_end(self):
        a = shp.geometry.LineString([(0, 0), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 5), (0, 4)])
        self.assertEqual((1.0, 0.0, 1.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_reverse_both(self):
        a = shp.geometry.LineString([(0, 0), (0, 5)])
        b = shp.geometry.LineString([(0, 1), (0, 0), (0, 5), (0, 4)])
        self.assertEqual((2.0, 0.0, 2.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_loop(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_loop_outside(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((3.0, 1.0, 4.0 / 9), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_loop_outside_reverse(self):
        a = shp.geometry.LineString([(0, 0), (0, 2), (1, 2), (1, 1), (0, 1), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 2), (1, 2), (2, 2), (2, 1), (2, 2), (2, 1), (1, 1), (0, 1), (0, 5)])
        self.assertEqual((5.0, 1.0, 6.0 / 9), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_loop_reverse(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 2), (0, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((4.0, 0.0, 4.0 / 9), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_missing_part(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((1.0, 3.0, 4.0 / 7), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_missing_late(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)])
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (1, 5)])
        self.assertEqual((3.0, 3.0, 6.0 / 6), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_parallel_same_direction(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)])
        self.assertEqual((5.0, 5.0, 10.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_parallel_opposite_direction(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString([(1, 5), (1, 4), (1, 2), (1, 1), (1, 0)])
        self.assertEqual((5.0, 5.0, 10.0 / 5), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_ab_zero(self):
        a = shp.geometry.LineString()
        b = shp.geometry.LineString()
        self.assertEqual((0.0, 0.0, 0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_a_zero(self):
        a = shp.geometry.LineString()
        b = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        self.assertEqual((5.0, 0.0, float('inf')), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_b_zero(self):
        a = shp.geometry.LineString([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        b = shp.geometry.LineString()
        self.assertEqual((0.0, 5.0, float('inf')), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_diagonal(self):
        a = shp.geometry.LineString([(0, 0), (1, 0), (3, 2), (3, 3)])
        b = shp.geometry.LineString([(0, 0), (1, 0), (2, 1), (3, 2), (3, 3)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_almost_equal(self):
        a = shp.geometry.LineString([(0, 0), (5, 0)])
        b = shp.geometry.LineString([(0.00000001, 0.00000002), (1.99999998, -0.00000001), (5.00000002, 0.00000001)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

    def test_compare_matches_diagonal_almost_equal(self):
        a = shp.geometry.LineString([(0, 0), (1, 0), (3, 2), (3, 3)])
        b = shp.geometry.LineString([(0, 0), (1.00000002, 0.00000001), (2.00000001, 1.00000001), (3.00000002, 2.00000002), (3, 3)])
        self.assertEqual((0.0, 0.0, 0.0), self.geodata.compare_matches(a, b, False))

if __name__ == '__main__':
    unittest.main()
