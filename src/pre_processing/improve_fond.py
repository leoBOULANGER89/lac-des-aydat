import os
import csv
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, linemerge, unary_union
from scipy.spatial.distance import cdist

from collections import defaultdict

import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


# ------------------------
# LECTURE DES CSV
# ------------------------
def read_curves(points_csv, lines_csv):
    pts = []
    with open(points_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            pts.append((float(row[0]), float(row[1]), float(row[2])))

    segs, depths = [], []
    with open(lines_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            segs.append((int(row[0]), int(row[1])))
            depths.append(float(row[2]))

    if len(pts) == 0 or len(segs) == 0:
        raise ValueError("Les fichiers de courbes sont vides")

    return np.array(pts), np.array(segs), np.array(depths)

# ------------------------ # EXPORT CSV # ------------------------
def save_point_cloud(points, output_csv): 
    pd.DataFrame(points, columns=["x","y","z"]).to_csv(output_csv, index=False)

# ------------------------
# CONSTRUCTION DES COURBES
# ------------------------
def curves_to_closed_loops(points, lines, depths):
    segments_by_depth = defaultdict(list)

    for (i, j), d in zip(lines, depths):
        seg = LineString([(points[i, 0], points[i, 1]),
                          (points[j, 0], points[j, 1])])
        segments_by_depth[d].append(seg)

    closed_loops = {}
    for depth, segments in segments_by_depth.items():
        merged = linemerge(unary_union(segments))
        loops = []

        if merged.geom_type == "LineString":
            geoms = [merged]
        else:
            geoms = list(merged.geoms)

        for g in geoms:
            if g.is_ring:
                loops.append(g)

        closed_loops[depth] = loops

    return closed_loops


import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union


# ------------------------
# UTILITAIRES GÉOMÉTRIQUES
# ------------------------

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def angle_between(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def local_tangent(coords, i):
    p_prev = coords[i - 1]
    p_next = coords[(i + 1) % len(coords)]
    return normalize(p_next - p_prev)


def local_normal(coords, i):
    t = local_tangent(coords, i)
    return np.array([-t[1], t[0]])


# ------------------------
# PROJECTION ORTHOGONALE
# ------------------------

def orthogonal_projection(point, curve):
    p = Point(point)
    proj_dist = curve.project(p)
    proj = curve.interpolate(proj_dist)
    return np.array([proj.x, proj.y])


# ------------------------
# HERMITE CUBIQUE
# ------------------------

def hermite_coeff(g0, g1, z0):
    a = g0 + g1
    b = -2*g0 - g1
    c = g0
    d = z0
    return a, b, c, d


def hermite_eval(a, b, c, d, t):
    return a*t**3 + b*t**2 + c*t + d


# ------------------------
# INTERPOLATION ENTRE DEUX COURBES
# ------------------------

def densify_same_level_curve(loop,
                             z,
                             angle_max=60,
                             step=1.0):

    coords = np.array(loop.coords)
    curve = LineString(coords)

    n = len(coords)

    # conserver points originaux
    output_pts = [[p[0], p[1], z] for p in coords]

    # pré-calcul normales
    normals = []
    for i in range(n):
        p_prev = coords[i - 1]
        p_next = coords[(i + 1) % n]
        t = p_next - p_prev
        t = t / (np.linalg.norm(t) + 1e-12)
        normals.append(np.array([-t[1], t[0]]))
    normals = np.array(normals)

    for i in range(n):

        P0 = coords[i]
        normal = normals[i]

        # projection sur la même courbe
        proj_dist = curve.project(Point(P0))
        P1 = np.array(curve.interpolate(proj_dist).coords[0])

        direction = P1 - P0
        length = np.linalg.norm(direction)

        if length < 1e-8:
            continue

        direction /= length

        ang = np.degrees(
            np.arccos(
                np.clip(np.dot(direction, normal), -1.0, 1.0)
            )
        )

        if ang > angle_max:
            continue

        # dz = 0 → surface plate
        g0 = 0.0
        g1 = 0.0

        a = g0 + g1
        b = -2*g0 - g1
        c = g0
        d = z

        n_samples = max(2, int(length / step))

        for j in range(1, n_samples):  # éviter doublons P0 et P1
            t = j / n_samples
            xy = (1 - t) * P0 + t * P1
            z_val = a*t**3 + b*t**2 + c*t + d
            output_pts.append([xy[0], xy[1], z_val])

    return np.array(output_pts)


def densify_all_same_levels(points_csv,
                            lines_csv,
                            output_csv,
                            angle_max=60,
                            step=1.0):

    points, lines, depths = read_curves(points_csv, lines_csv)
    loops_by_depth = curves_to_closed_loops(points, lines, depths)

    all_points = []

    for z, loops in loops_by_depth.items():
        for loop in loops:
            pts = densify_same_level_curve(
                loop,
                z,
                angle_max=angle_max,
                step=step
            )
            all_points.append(pts)

    all_points = np.vstack(all_points)
    save_point_cloud(all_points, output_csv)

    print("Points densifiés (même niveau uniquement).")


# ------------------------
# EXECUTION
# ------------------------
if __name__ == "__main__":
    densify_all_same_levels(
        "../donnee/curves/Lake_Aydat_points.csv",
        "../donnee/curves/Lake_Aydat_lines.csv",
        "../donnee/point_cloud/Lake_Aydat_better.csv"
    )