#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (C) 2020-2021 Adrian Wöltche
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

import sys
import uuid

import lib.tracks

crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
reproject_crs = "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

print("Loading floating car data...", end=" ")
tracks = lib.tracks.Tracks("data/points_original.csv", delimiter=';', groupby=['device', 'subid'], reload=True, crs=crs, reproject_crs=reproject_crs)
print("done")

anonymize_buffer = 300

csv = []
ids = set()

for name in tracks.points_group.groups.keys():
    print("Anonymizing {} ...".format(name))
    track_points, _ = tracks.get_track(name, anonymize=True, anonymize_buffer=anonymize_buffer)

    if len(track_points) <= 0:
        print("Track too short after anonymization, skipping...")
        continue

    points = track_points.to_crs(crs=crs)
    points.insert(4, 'lon', points['geometry'].x.apply(lambda x: round(x, 6)))
    points.insert(5, 'lat', points['geometry'].y.apply(lambda x: round(x, 6)))
    points = points.drop(columns=['geometry'])

    id = uuid.uuid4()
    while id in ids:
        id = uuid.uuid4()
    ids.add(id)

    points['device'] = id
    points['subid'] = 0

    csv.append(points.to_csv(index=False, sep=';', header=len(csv) == 0))

with open("data/points_anonymized.csv", 'w') as file:
    file.write(''.join(csv))
