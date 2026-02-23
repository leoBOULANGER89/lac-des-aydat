import os
import pandas as pd
import pyvista as pv

# ======================
# PARAMÈTRES
# ======================
input_csv = "../donnee/point_cloud/Lake_Aydat.csv"
output_dir = "../resultat/Lake_Aydat/"
output_obj = os.path.join(output_dir, "Lake_Aydat_Poisson_like.obj")

os.makedirs(output_dir, exist_ok=True)

# ======================
# LIRE LE NUAGE DE POINTS
# ======================
df = pd.read_csv(input_csv)
points = df[['x', 'y', 'z']].values

cloud = pv.PolyData(points)

# ======================
# TRIANGULATION 2D (plan XY)
# ======================
# max_edge_length peut être ajusté pour supprimer les triangles trop grands
mesh = cloud.delaunay_2d(alpha=1000.0)

# ======================
# CALCUL DES NORMALES
# ======================
mesh = mesh.compute_normals(cell_normals=True, point_normals=False)

# ======================
# SAUVEGARDE EN OBJ
# ======================
mesh.save(output_obj)

print(f"✅ Surface Poisson-like générée : {output_obj}")
