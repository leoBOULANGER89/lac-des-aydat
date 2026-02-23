#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extraction et approximation de courbes de niveau à partir d'une image bathymétrique.

Description
-----------
Ce module permet de reconstruire des courbes de niveau (isobathes)
à partir d'une image RGB traitée où chaque couleur représente une hauteur.

Le pipeline complet est le suivant :
    1. Lecture d'une image bathymétrique traitée.
    2. Lecture d'un fichier CSV de légende associant couleurs et hauteurs.
    3. Conversion de l'image en carte de hauteur (depth map).
    4. Extraction des courbes de niveau via la méthode des Marching Squares
       (skimage.measure.find_contours).
    5. Approximation des courbes par un échantillonnage métrique régulier.
    6. Reconstruction optionnelle des segments reliant les points.
    7. Export des résultats au format CSV.

Fonctionnalités principales
---------------------------
- Conversion pixel → coordonnées métriques (mètre/pixel).
- Association robuste couleur → hauteur (distance RGB minimale).
- Échantillonnage métrique contrôlé des contours.
- Reconstruction des boucles fermées via Shapely.
- Export :
    • nuage de points 3D (x, y, z)
    • segments reliant les points (indices)

Entrées attendues
-----------------
Image traitée (PNG, JPG, etc.)
    Image RGB où chaque couleur correspond à une hauteur.

Fichier de légende (CSV)
    Format attendu :
        - Première ligne : facteur d'échelle (mètre/pixel)
        - Deuxième ligne : en-tête
        - Lignes suivantes : couleur hexadécimale ; hauteur

Exemple :
    scale;0.5
    color;depth
    #0000FF;-5
    #00FFFF;-10

Sorties générées
----------------
Points (CSV)
    Colonnes : x, y, z
    Coordonnées métriques, hauteur signée.

Segments (optionnel)
    Colonnes : loop_id, ix, iy, depth
    - ix, iy : indices des points reliés
    - loop_id : identifiant de la boucle
        -1 = segment non fermé ou point isolé

Paramètres CLI
--------------
image_name : str
    Nom de l'image traitée.
--pas : float
    Pas métrique d'échantillonnage (en mètres).
--p :
    Mode points uniquement (désactive la génération des segments).

Notes techniques
----------------
- Les hauteurs sont retournées avec le signe défini dans la légende.
- Les coordonnées Y sont inversées pour passer du repère image
  au repère cartésien métrique.
- La méthode Marching Squares fournit des contours sous-pixel.
- L'échantillonnage métrique permet de contrôler la densité des points
  indépendamment de la résolution de l'image.
- Les dossiers de sortie sont créés automatiquement si nécessaires.

Cas d’usage typique
-------------------
Préparation de données pour :
    - triangulation Delaunay / CDT,
    - reconstruction de surface 3D,
    - génération de maillages,
    - interpolation bathymétrique,
    - export vers SIG ou moteur 3D.

Auteur
------
Projet de reconstruction bathymétrique à partir d'image traitée.
"""

import os
import csv
import numpy as np
from PIL import Image
from skimage.measure import find_contours

from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from ..io import RAndW
from collections import defaultdict

# =============================================================================================================
# CHECK
# =============================================================================================================
def check_files_exist(INPUT_IMAGE, INPUT_CSV):
    """
    Vérifie que les fichiers image et CSV existent.
    Lève FileNotFoundError si l'un des fichiers est manquant.
    """
    if not os.path.isfile(INPUT_IMAGE):
        raise FileNotFoundError(f"Le fichier image n'existe pas : {INPUT_IMAGE}")
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Le fichier CSV n'existe pas : {INPUT_CSV}")




# =============================================================================================================
# PARAMÈTRES
# =============================================================================================================
def determine_param (image_name, points_only):
    """
    Détermine les chemins des fichiers d'entrée et de sortie
    à partir du nom d'une image traitée.

    Cette fonction construit automatiquement :
    - le chemin vers l'image d'entrée
    - le fichier CSV de légende associé
    - les fichiers CSV de sortie contenant les points et les lignes extraits

    Parameters
    ----------
    image_name : str
        Nom du fichier image à traité (ex: "carte_traitee.png").

    Returns
    -------
    INPUT_IMAGE : str
        Chemin vers l'image d'entrée.
    INPUT_CSV : str
        Chemin vers le fichier CSV de légende.
    OUTPUT_POINTS : str
        Chemin vers le fichier CSV de sortie des points.
    OUTPUT_LINES : str
        Chemin vers le fichier CSV de sortie des lignes.
    """

    data_name = image_name.replace("_traitee.png","")
    data_path = "data/raw/map/" + data_name + "/"
    INPUT_IMAGE = data_path + image_name
    INPUT_CSV = data_path + "légende.csv"
    

    if points_only:
        output_dir = "data/processed/point_cloud/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_POINTS = output_dir + data_name + ".csv"
        OUTPUT_LINES  = None
    else:
        output_dir = "data/processed/curves/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_POINTS = output_dir + data_name + "_points.csv"
        OUTPUT_LINES  = output_dir + data_name + "_lines.csv"
        

    return INPUT_IMAGE, INPUT_CSV, OUTPUT_POINTS, OUTPUT_LINES

# =============================================================================================================
# OUTILS
# =============================================================================================================
def hex_to_rgb(hex_color):
    """
    Convertit une couleur hexadécimale en tuple RGB.

    Parameters
    ----------
    hex_color : str
        Couleur au format hexadécimal (ex: "#FF00AA" ou "FF00AA").

    Returns
    -------
    tuple of int
        Tuple (R, G, B) avec des valeurs comprises entre 0 et 255.
    """

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def dist(p0, p1):
    """
    Calcule la distance euclidienne entre deux points 2D.

    Parameters
    ----------
    p0 : tuple or array-like
        Premier point (x, y).
    p1 : tuple or array-like
        Second point (x, y).

    Returns
    -------
    float
        Distance euclidienne entre p0 et p1.
    """

    return np.hypot(p1[0] - p0[0], p1[1] - p0[1])


# =============================================================================================================
# LECTURE CSV LÉGENDE
# =============================================================================================================
def lecture_legende(INPUT_CSV):
    """
    Lit un fichier CSV de légende associant des couleurs à des hauteurs.

    Le fichier CSV doit contenir :
    - une première ligne indiquant l'échelle (scale : mètre/pixel)
    - une ligne d'en-tête
    - des lignes associant une couleur hexadécimale à une hauteur

    Parameters
    ----------
    INPUT_CSV : str
        Chemin vers le fichier CSV de légende.

    Returns
    -------
    scale : float
        Facteur d'échelle extrait de la première ligne du fichier mètre/pixel.
    depth_values : list of float
        Liste triée des valeurs de hauteur.
    color_depth : dict
        Dictionnaire associant des couleurs RGB (tuple) aux hauteurs.
    """
        
    color_depth = {}
    depth_values = []

    try:
        with open(INPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader)
            scale = float(first_row[1])
            next(reader)
            for row in reader:
                if len(row) < 2:
                    raise ValueError(f"Ligne mal formatée dans {INPUT_CSV}: {row}")
                rgb = hex_to_rgb(row[0])
                depth = float(row[1])
                color_depth[rgb] = depth
                depth_values.append(depth)
        depth_values = sorted(depth_values)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier CSV {INPUT_CSV} : {e}")

    return scale, depth_values, color_depth

# =============================================================================================================
# LECTURE IMAGE
# =============================================================================================================
def lecture_image(INPUT_IMAGE):
    """
    Lit une image RGB et la convertit en tableau NumPy.

    Parameters
    ----------
    INPUT_IMAGE : str
        Chemin vers le fichier image.

    Returns
    -------
    img_array : numpy.ndarray
        Tableau de forme (height, width, 3) contenant les valeurs RGB.
    height : int
        Hauteur de l'image en pixels.
    width : int
        Largeur de l'image en pixels.
    """

    img = Image.open(INPUT_IMAGE).convert("RGB")
    img_array = np.array(img)
    height, width, _ = img_array.shape

    return img_array, height, width

# =============================================================================================================
# CARTE DE PROFONDEUR
# =============================================================================================================
def array_image_to_depth_map (img_array, height, width, color_depth):
    """
    Convertit une image RGB en carte de hauteur.

    Chaque pixel est associé à une hauteur en fonction
    de la couleur de référence la plus proche.

    Parameters
    ----------
    img_array : numpy.ndarray
        Tableau RGB de l'image (height, width, 3).
    height : int
        Hauteur de l'image en pixels.
    width : int
        Largeur de l'image en pixels.
    color_depth : dict
        Dictionnaire associant des couleurs RGB à des hauteurs.

    Returns
    -------
    depth_map : numpy.ndarray
        Carte de hauteur de forme (height, width).
    """
    # ----------------------------------------------------------------
    # légende vide
    # ----------------------------------------------------------------
    if not color_depth:
        print("⚠ Aucune couleur de référence dans la légende")
        return np.full((height, width), np.nan), np.nan

    # ----------------------------------------------------------------
    # préparation
    # ----------------------------------------------------------------
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    ref_colors = np.array(list(color_depth.keys()), dtype=np.float32)
    depths = np.array(list(color_depth.values()), dtype=np.float32)

    # ----------------------------------------------------------------
    # distances RGB
    # ----------------------------------------------------------------
    diff = pixels[:, None, :] - ref_colors[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)

    min_idx = np.argmin(dist2, axis=1)
    min_dist = np.sqrt(dist2[np.arange(dist2.shape[0]), min_idx])

    # ----------------------------------------------------------------
    # hauteur associée
    # ----------------------------------------------------------------
    depth_map = depths[min_idx].reshape(height, width)

    return depth_map

# =============================================================================================================
# EXTRACTION
# =============================================================================================================
def assign_loop_ids(points_list, lines_list):
    """
    Assigne un ID à chaque boucle fermée et retourne les segments
    en termes d'indices de points au lieu de coordonnées.

    Parameters
    ----------
    points_list : list of (x, y, depth)
        Liste des points avec leurs coordonnées et hauteur.
    lines_list : list of (i0, i1, depth)
        Liste des segments reliant deux indices de points à une hauteur.

    Returns
    -------
    result : list of (loop_id, i0, i1, depth)
        Liste de tous les segments avec l'ID de la boucle.
        loop_id = -1 pour les segments non fermés ou points isolés.
    """
     # Regrouper les segments par hauteur
    segments_by_depth = defaultdict(list)
    for i0, i1, d in lines_list:
        segments_by_depth[d].append((i0, i1))

    result = []
    loop_id_counter = 0

    for depth, segs in segments_by_depth.items():
        if not segs:
            continue

        # Construire les LineStrings en utilisant les indices
        line_strings = [LineString([(points_list[i0][0], points_list[i0][1]),
                                    (points_list[i1][0], points_list[i1][1])])
                        for i0, i1 in segs]

        merged = linemerge(unary_union(line_strings))

        # Normaliser en liste
        if merged.geom_type == "LineString":
            geoms = [merged]
        else:
            geoms = list(merged.geoms)

        # Marquer les indices utilisés
        points_in_loops = set()

        for g in geoms:
            coords = list(g.coords)
            # Créer un mapping coord -> index pour cette hauteur
            coord_to_idx = { (points_list[i][0], points_list[i][1]): i for i0, i1 in segs for i in (i0,i1) }

            if g.is_ring:
                # boucle fermée
                for i in range(len(coords)-1):
                    i0 = coord_to_idx[coords[i]]
                    i1 = coord_to_idx[coords[i+1]]
                    result.append((loop_id_counter, i0, i1, depth))
                    points_in_loops.update([i0, i1])
                loop_id_counter += 1
            else:
                # segment non fermé
                for i in range(len(coords)-1):
                    i0 = coord_to_idx[coords[i]]
                    i1 = coord_to_idx[coords[i+1]]
                    result.append((-1, i0, i1, depth))
                    points_in_loops.update([i0, i1])

        # Ajouter les points isolés
        for i, (_, _, d) in enumerate(points_list):
            if d == depth and i not in points_in_loops:
                result.append((-1, i, i, depth))

    return result


def extraction(depth_values, scale, pas_m, depth_map, height, make_curves=True):
    """
    Extrait des courbes de niveau et génère les segments.
    """

    points_list = []
    lines_tmp = [] if make_curves else None
    point_id_counter = 0

    for level in depth_values:

        contours = find_contours(depth_map, level)

        for contour in contours:

            if len(contour) < 2:
                continue

            start_point_id = None
            prev_kept_id = None
            prev_kept_point = None
            dist_acc = 0.0

            for i in range(len(contour) - 1):

                y0, x0 = contour[i]
                y1, x1 = contour[i + 1]

                p0 = (x0 * scale, (height - y0) * scale)
                p1 = (x1 * scale, (height - y1) * scale)

                seg_len = dist(p0, p1)
                dist_acc += seg_len

                # Premier point
                if prev_kept_point is None:
                    points_list.append([p0[0], p0[1], level])
                    prev_kept_id = point_id_counter
                    start_point_id = point_id_counter
                    point_id_counter += 1
                    prev_kept_point = p0
                    continue

                # Échantillonnage métrique
                if dist_acc >= pas_m:
                    points_list.append([p1[0], p1[1], level])
                    new_id = point_id_counter
                    point_id_counter += 1

                    if make_curves:
                        lines_tmp.append([prev_kept_id, new_id, level])

                    prev_kept_id = new_id
                    prev_kept_point = p1
                    dist_acc = 0.0

            # Fermeture du contour
            if make_curves and prev_kept_id is not None and start_point_id is not None:
                lines_tmp.append([prev_kept_id, start_point_id, level])

    # Reconstruction des boucles fermées
    if make_curves:
        lines_list = assign_loop_ids(points_list, lines_tmp)
        return points_list, lines_list

    return points_list, None

# ======================
# MAIN
# ======================
def traitement_main(
    INPUT_IMAGE,
    INPUT_CSV,
    OUTPUT_POINTS,
    pas_m=100.0,
    points_only=False,
    OUTPUT_LINES=None,
):
    """
        Fonction principale de traitement des courbes de niveau.

    Cette fonction :
    - lit la légende (échelle + hauteurs)
    - lit l'image traitée
    - génère la carte de hauteur
    - extrait les courbes de niveau
    - sauvegarde les résultats dans des fichiers CSV

    L'extraction et la sauvegarde des segments (lignes) peuvent
    être désactivées afin de ne conserver que les points.

    Parameters
    ----------
    INPUT_IMAGE : str
        Chemin vers le fichier image traité.
    INPUT_CSV : str
        Chemin vers le fichier CSV de légende.
    OUTPUT_POINTS : str
        Chemin du fichier CSV de sortie des points.
    pas_m : float, optional
        Pas métrique d'échantillonnage des courbes (en mètres).
    points_only : bool, optional
        Si True, extrait et sauvegarde uniquement les points.
    OUTPUT_LINES : str or None, optional
        Chemin du fichier CSV de sortie des segments.
        Obligatoire si points_only = False.

    Returns
    -------
    points_list : list
        Liste des points extraits [x, y, z].
    lines_list : list or None
        Liste des segments extraits [ix, iy, depth] si points_only=False,
        sinon None.
    """

   
    # checks
    check_files_exist(INPUT_IMAGE, INPUT_CSV)

    # lecture légende
    scale, depth_values, color_depth = lecture_legende(INPUT_CSV)

    # lecture image
    img_array, height, width = lecture_image(INPUT_IMAGE)

    # carte de hauteur
    depth_map = array_image_to_depth_map(
        img_array, height, width, color_depth
    )

    # extraction
    points_list, lines_list = extraction(
        depth_values,
        scale,
        pas_m,
        depth_map,
        height,
        make_curves=not points_only
    )

    # sauvegarde
    if points_only:
        RAndW.save_CSV(
            OUTPUT_POINTS,
            points_list,
            save_lines=False
        )
    else:
        if OUTPUT_LINES is None:
            raise ValueError("OUTPUT_LINES doit être fourni si points_only=False")

        RAndW.save_CSV(
            OUTPUT_POINTS,
            points_list,
            OUTPUT_LINES,
            lines_list,
            save_lines=True
        )

    return points_list, lines_list



if __name__ == "__main__":
    import argparse

    # ----------------------------------------------------------------
    # ARGUMENTS CLI
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Approximation des courbes de niveaux par des droites"
    )

    parser.add_argument(
        "image_name",
        type=str,
        nargs="?",
        default="Lake_Aydat_traitee.png",
        help="Nom de l'image traitée (ex: Lake_Aydat_traitee.png)"
    )

    parser.add_argument(
        "--pas",
        type=float,
        default=10.0,
        help="Pas métrique d'échantillonnage des courbes (en mètres)"
    )

    parser.add_argument(
        "--p",
        action="store_true",
        help="Extraire uniquement les points (sans générer les courbes)"
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # PARAMÈTRES
    # ----------------------------------------------------------------
    image_name = args.image_name
    pas_m = args.pas
    points_only = args.p

    name = image_name.replace("_traitee.png", "")

    print("Approximation des courbes de niveaux par des droites :")
    print(f"\tPas d'échantillonnage : {pas_m} m")
    print(f"\tMode : {'points uniquement' if points_only else 'points + courbes'}")

    # ----------------------------------------------------------------
    # PIPELINE AVEC GESTION D'ERREURS
    # ----------------------------------------------------------------
    try:
        # chemins
        print(f"\tRécupération des fichiers pour {name}")
        INPUT_IMAGE, INPUT_CSV, OUTPUT_POINTS, OUTPUT_LINES = determine_param(image_name, points_only)

        # lecture légende
        if not os.path.isfile(INPUT_IMAGE):
            raise FileNotFoundError(f"Le fichier image n'existe pas : {INPUT_IMAGE}")
        if not os.path.isfile(INPUT_CSV):
            raise FileNotFoundError(f"Le fichier CSV n'existe pas : {INPUT_CSV}")

        print(f"\tLecture de la légende : {INPUT_CSV}")
        scale, depth_values, color_depth = lecture_legende(INPUT_CSV)

        # lecture image
        print(f"\tLecture de l'image : {INPUT_IMAGE}")
        img_array, height, width = lecture_image(INPUT_IMAGE)

        # construction carte de hauteur
        print("\tConstruction de la carte de hauteur")
        depth_map = array_image_to_depth_map(img_array, height, width, color_depth)

        # extraction des données
        print("\tExtraction des données")
        points_list, lines_list = extraction(
            depth_values,
            scale,
            pas_m,
            depth_map,
            height,
            make_curves=not points_only
        )

        # sauvegarde
        print("\tSauvegarde des résultats")
        

        RAndW.save_CSV(
            OUTPUT_POINTS,
            points_list,
            OUTPUT_LINES,
            lines_list,
            save_lines=not points_only
        )   

        # résumé
        print(f"\t✅ Points enregistrés : {OUTPUT_POINTS}")
        print(f"\tNombre de points : {len(points_list)}")

        if not points_only:
            print(f"\t✅ Droites enregistrées : {OUTPUT_LINES}")
            print(f"\tNombre de droites : {len(lines_list)}")

    except FileNotFoundError as fnf_err:
        print(f"❌ Fichier manquant : {fnf_err}")
    except ValueError as val_err:
        print(f"❌ Erreur de format : {val_err}")
    except Exception as e:
        print(f"❌ Une erreur inattendue est survenue : {e}")
