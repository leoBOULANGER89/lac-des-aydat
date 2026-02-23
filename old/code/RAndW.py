import os
import csv
import numpy as np
from PIL import Image
from skimage.measure import find_contours
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, linemerge, unary_union
from scipy.spatial.distance import cdist

# =============================================================================================================
# CHECK
# =============================================================================================================
def check_files_exist(INPUT_FILES):
    """
    Vérifie que les fichiers.
    Lève FileNotFoundError si l'un des fichiers est manquant.
    """
    for file in INPUT_FILES:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Le fichier image n'existe pas : {file}")
        

# =============================================================================================================
# CSV
# =============================================================================================================

# ------------------------
# LECTURE CSV
# ------------------------
def read_curves(INPUT_CSV_POINTS, INPUT_CSV_LINES):
    pts = []
    with open(INPUT_CSV_POINTS, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            pts.append((float(row[0]), float(row[1]), float(row[2])))



    lines = []
    depths = []

    with open(INPUT_CSV_LINES, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)

        seg = []
        depth = None
        old_id = None

        for row in reader:
            current_id = int(row[0])

            if old_id is not None and old_id != current_id:
                lines.append(seg)
                depths.append(depth)
                seg = []

            seg.append((int(row[1]), int(row[2])))
            depth = float(row[3])

            old_id = current_id
        
        if seg:
            lines.append(seg)
            depths.append(depth)

    if len(pts) == 0 or len(lines) == 0:
        raise ValueError("Les fichiers de courbes sont vides")

    return np.array(pts), np.array(lines), np.array(depths)

def read_points (INPUT_CSV_POINTS):
    """
    Lit un fichier CSV contenant des coordonnées 3D et extrait
    les points sous forme de tableau numpy.

    Le fichier CSV doit contenir les colonnes : x, y, z.

    Parameters
    ----------
    INPUT_CSV_POINTS : str
        Chemin vers le fichier CSV contenant le nuage de point.

    Returns
    -------
    points : numpy.ndarray
        Tableau de forme (N, 3) contenant les coordonnées (x, y, z).
    """

    data = pd.read_csv(INPUT_CSV_POINTS)
    points = data[["x", "y", "z"]].values
    return points

# ------------------------
# ÉCRITURE CSV
# ------------------------
def save_CSV(OUTPUT_POINTS, points_list, OUTPUT_LINES = None, lines_list=None, save_lines=False):
    """
    Sauvegarde les points (et optionnellement les segments)
    des courbes de niveau dans des fichiers CSV.

    Par défaut, les points et les segments sont sauvegardés.
    Il est possible de ne sauvegarder que les points en
    désactivant l'écriture des lignes.

    Parameters
    ----------
    OUTPUT_POINTS : str
        Chemin du fichier CSV contenant les points (x, y, z).
    points_list : list of list
        Liste des points [x, y, z].
    OUTPUT_LINES : str or None
        Chemin du fichier CSV contenant les segments
        (index_point_1, index_point_2, profondeur).
    lines_list : list of list or None, optional
        Liste des segments [id, ix, iy, depth].
        Ignorée si save_lines = False.
    save_lines : bool, optional
        Indique si les segments doivent être sauvegardés.
        Par défaut True.

    Returns
    -------
    None
    """

    # sauvegarde des points
    with open(OUTPUT_POINTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points_list)

    # sauvegarde optionnelle des lignes
    if save_lines:
        if OUTPUT_LINES is None or lines_list is None:
            raise ValueError("OUTPUT_LINES et lines_list doivent être fournis si save_lines=True")

        with open(OUTPUT_LINES, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "ix", "iy", "depth"])
            writer.writerows(lines_list)
