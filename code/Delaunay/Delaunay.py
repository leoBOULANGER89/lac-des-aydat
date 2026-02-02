import os
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# -----------------------------
# PARAMÈTRES
# -----------------------------

data_path = "../../donnee/point_cloud/"
csv_name = "Lake_Aydat.csv"
csv_path = data_path + csv_name
output_dir = "../../resultat/" + csv_name.replace(".csv", "") + "/"

os.makedirs(output_dir, exist_ok=True)
output_dir += "Delaunay/"
os.makedirs(output_dir, exist_ok=True)

output_obj = output_dir + csv_name.replace(".csv", "") + "_Delaunay.obj"


# ===== LECTURE DU CSV =====
data = pd.read_csv(csv_path)
points = data[["x", "y", "z"]].values

# ===== TRIANGULATION (sur x,y) =====
tri = Delaunay(points[:, :2])

# ===== ÉCRITURE DU OBJ =====
with open(output_obj, "w") as f:
    f.write("# OBJ généré depuis un CSV (x,y,z)\n")

    # Sommets
    for p in points:
        f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    # Faces (indices OBJ commencent à 1)
    for simplex in tri.simplices:
        i, j, k = simplex + 1
        f.write(f"f {i} {j} {k}\n")

print(f"Export terminé : {output_obj}")