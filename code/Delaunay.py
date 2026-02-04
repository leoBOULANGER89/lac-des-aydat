import os
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# -----------------------------
# PARAMÈTRES
# -----------------------------
name = "Lake_Aydat"


data_path = "../donnee/point_cloud/"
csv_path = data_path + name + ".csv"
output_dir = "../../resultat/" + name + "/"
os.makedirs(output_dir, exist_ok=True)

output_obj = output_dir + name + "_Delaunay.obj"


# -----------------------------
# LECTURE DU CSV
# -----------------------------

data = pd.read_csv(csv_path)
points = data[["x", "y", "z"]].values

# -----------------------------
# DELAUNAY
# -----------------------------

tri = Delaunay(points[:, :2])

# -----------------------------
# ÉCRITURE DU OBJ
# -----------------------------

with open(output_obj, "w") as f:
    f.write("# OBJ généré depuis un CSV (x,y,z)\n")

    # Sommets
    for p in points:
        f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    # Faces (indices OBJ commencent à 1)
    for simplex in tri.simplices:
        i, j, k = simplex + 1
        f.write(f"f {i} {j} {k}\n")

print(f"✅ Surface par Delaunay enregistré : {output_obj}")