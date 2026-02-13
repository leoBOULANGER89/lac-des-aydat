#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained Delaunay Triangulation (CDT) à partir d'un nuage de points
et de segments de courbes. Génère un fichier OBJ.
"""

import os
import csv
import numpy as np
import triangle as tr

# =====================================
# PARAMÈTRES
# =====================================
name = "Lake_Aydat"
points_csv = f"../donnee/curves/{name}_points.csv"
lines_csv  = f"../donnee/curves/{name}_lines.csv"
output_dir = f"../resultat/{name}/"
os.makedirs(output_dir, exist_ok=True)
output_obj = os.path.join(output_dir, f"{name}_CDT.obj")

# =====================================
# LECTURE POINTS
# =====================================
points_list = []
with open(points_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        points_list.append([float(row["x"]), float(row["y"]), float(row["z"])])

points_array = np.array([[x, y] for x, y, z in points_list])
z_values     = np.array([z for x, y, z in points_list])

# =====================================
# LECTURE SEGMENTS (CONTRAINTES)
# =====================================
segments_list = []
with open(lines_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        segments_list.append([int(row["ix"]), int(row["iy"])])

segments_array = np.array(segments_list)

# =====================================
# PRÉPARATION TRIANGLE
# =====================================
A = dict(vertices=points_array, segments=segments_array)

# 'p' = Planar Straight Line Graph (respect des segments)
# 'q30' = angle minimal 30° pour qualité des triangles
B = tr.triangulate(A, 'pq30a1000D')

triangles = B['triangles']
vertices  = B['vertices']

# =====================================
# CONSTRUCTION Z PAR INTERPOLATION
# =====================================

# Associer z aux sommets initiaux
original_xy = points_array
original_z  = z_values

# Si Triangle a gardé les sommets originaux au début (ce qu'il fait)
# Les premiers len(original_xy) vertices correspondent aux points d'entrée
new_vertices = vertices
new_z = np.zeros(len(new_vertices))

# Copier les z des sommets d'origine
new_z[:len(original_z)] = original_z

# Pour les points Steiner → interpolation barycentrique
from scipy.spatial import Delaunay

# Triangulation des points d'origine seulement
tri_orig = Delaunay(original_xy)

for i in range(len(original_z), len(new_vertices)):
    pt = new_vertices[i]
    simplex = tri_orig.find_simplex(pt)
    
    if simplex >= 0:
        verts = tri_orig.simplices[simplex]
        transform = tri_orig.transform[simplex]
        bary = np.dot(transform[:2], pt - transform[2])
        bary = np.append(bary, 1 - bary.sum())
        new_z[i] = np.dot(bary, original_z[verts])
    else:
        # fallback si hors domaine
        dists = np.hypot(original_xy[:,0]-pt[0], original_xy[:,1]-pt[1])
        new_z[i] = original_z[np.argmin(dists)]



# =====================================
# SAUVEGARDE OBJ
# =====================================
with open(output_obj, "w") as f:
    for i, v in enumerate(new_vertices):
        f.write(f"v {v[0]} {v[1]} {new_z[i]}\n")
    
    for tri in triangles:
        i, j, k = tri + 1
        f.write(f"f {i} {j} {k}\n")

print(f"✅ CDT OBJ généré : {output_obj}")
