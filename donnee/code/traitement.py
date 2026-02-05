#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extraction et approximation de courbes de niveau à partir d'une image traitée.

Ce programme permet de :
- lire une image bathymétrique traitée (couleurs représentant des profondeurs),
- lire un fichier CSV de légende associant couleurs et profondeurs,
- reconstruire une carte de profondeur à partir de l'image,
- extraire des courbes de niveau à l'aide de la méthode de Marching Squares,
- approximer ces courbes par des segments de droites,
- exporter les résultats sous forme de fichiers CSV.

Deux types de sorties peuvent être générés :
- un fichier CSV contenant uniquement les points 3D (x, y, z),
- un fichier CSV supplémentaire contenant les segments reliant ces points
  (représentation linéaire des courbes).

L'utilisateur peut choisir d'extraire uniquement les points ou bien
les points et les courbes associées.

Entrées attendues
-----------------
Image traitée :
    Image RGB (ex : PNG) où chaque couleur correspond à une profondeur donnée.

Fichier de légende :
    CSV contenant :
    - une première ligne avec le facteur d'échelle (mètre / pixel),
    - une ligne d'en-tête,
    - des lignes associant une couleur hexadécimale à une profondeur.

Sorties générées
----------------
Points :
    CSV contenant les colonnes : x, y, z

Courbes (optionnel) :
    CSV contenant les colonnes : ix, iy, depth
    où ix et iy sont les indices des points reliés.

Utilisation
-----------
python extraction_courbes.py
python extraction_courbes.py Lake_Aydat_traitee.png
python extraction_courbes.py Lake_Aydat_traitee.png --pas 50
python extraction_courbes.py Lake_Aydat_traitee.png --p

Notes
-----
- Les coordonnées sont exprimées dans un repère métrique (conversion pixel → mètre).
- Les profondeurs sont retournées avec un signe négatif (bathymétrie).
- Les dossiers de sortie sont créés automatiquement s'ils n'existent pas.
"""

import os
import csv
import numpy as np
from PIL import Image
from skimage.measure import find_contours

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
def determine_param (image_name):
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
    data_path = "../raw/map/" + data_name + "/"
    INPUT_IMAGE = data_path + image_name
    INPUT_CSV = data_path + "légende.csv"
    output_dir = "../curves/"
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
    Lit un fichier CSV de légende associant des couleurs à des profondeurs.

    Le fichier CSV doit contenir :
    - une première ligne indiquant l'échelle (scale : mètre/pixel)
    - une ligne d'en-tête
    - des lignes associant une couleur hexadécimale à une profondeur

    Les profondeurs sont retournées avec un signe négatif
    (ex : bathymétrie).

    Parameters
    ----------
    INPUT_CSV : str
        Chemin vers le fichier CSV de légende.

    Returns
    -------
    scale : float
        Facteur d'échelle extrait de la première ligne du fichier mètre/pixel.
    depth_values : list of float
        Liste triée des valeurs de profondeur.
    color_depth : dict
        Dictionnaire associant des couleurs RGB (tuple) aux profondeurs.
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
                depth = -float(row[1])
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
    Convertit une image RGB en carte de profondeur.

    Chaque pixel est associé à une profondeur en fonction
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
        Dictionnaire associant des couleurs RGB à des profondeurs.

    Returns
    -------
    depth_map : numpy.ndarray
        Carte de profondeur de forme (height, width).
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
    # profondeur associée
    # ----------------------------------------------------------------
    depth_map = depths[min_idx].reshape(height, width)

    return depth_map

# =============================================================================================================
# EXTRACTION
# =============================================================================================================
def extraction(depth_values, scale, pas_m, depth_map, height, make_curves=True):
    """
    Extrait des courbes de niveau à partir d'une carte de profondeur
    et les approxime par des segments de droites.

    Pour chaque niveau de profondeur :
    - les contours sont détectés avec la méthode de Marching Squares
      (via 'skimage.measure.find_contours')
    - les points sont échantillonnés régulièrement selon un pas métrique
    - les segments successifs sont générés optionnellement

    Les coordonnées sont exprimées dans un repère métrique
    (conversion pixel → mètre).

    Parameters
    ----------
    depth_values : list of float
        Liste des niveaux de profondeur à extraire.
    scale : float
        Facteur d'échelle (mètre / pixel).
    pas_m : float
        Pas d'échantillonnage des courbes de niveau.
    depth_map : numpy.ndarray
        Carte de profondeur 2D.
    height : int
        Hauteur de l'image en pixels (pour inversion de l'axe Y).
    make_curves : bool, optional
        Indique si les segments de courbes doivent être générés.
        Par défaut True.

    Returns
    -------
    points_list : list of list [x, y, depth]
        Liste des points extraits (coordonnées métriques).
    lines_list : list of list [i0, i1, depth] or None
        Liste des segments reliant deux points avec leur profondeur
        associée si make_curves=True, sinon None.
    """

    points_list = []
    lines_list = [] if make_curves else None
    point_id_counter = 0

    for level in depth_values:
        contours = find_contours(depth_map, level)

        for contour in contours:
            if len(contour) < 2:
                continue

            start_point_id = None
            prev_kept_point = None
            prev_kept_id = None
            dist_acc = 0.0

            for i in range(len(contour) - 1):
                y0, x0 = contour[i]
                y1, x1 = contour[i + 1]

                p0 = (x0 * scale, (height - y0) * scale)
                p1 = (x1 * scale, (height - y1) * scale)

                seg_len = dist(p0, p1)
                dist_acc += seg_len

                if prev_kept_point is None:
                    # premier point du contour
                    points_list.append([p0[0], p0[1], level])
                    prev_kept_id = point_id_counter
                    start_point_id = point_id_counter
                    point_id_counter += 1
                    prev_kept_point = p0
                    continue

                if dist_acc >= pas_m:
                    # on garde p1 tel quel
                    points_list.append([p1[0], p1[1], level])
                    new_id = point_id_counter
                    point_id_counter += 1

                    if make_curves:
                        lines_list.append([prev_kept_id, new_id, level])

                    prev_kept_point = p1
                    prev_kept_id = new_id
                    dist_acc = 0.0

            # fermeture du contour
            if make_curves and prev_kept_id is not None and start_point_id is not None:
                lines_list.append([prev_kept_id, start_point_id, level])

    return points_list, lines_list
        

# =============================================================================================================
# ÉCRITURE CSV
# =============================================================================================================
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
        Liste des segments [ix, iy, depth].
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
            writer.writerow(["ix", "iy", "depth"])
            writer.writerows(lines_list)


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
    - lit la légende (échelle + profondeurs)
    - lit l'image traitée
    - génère la carte de profondeur
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

    # carte de profondeur
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
        save_CSV(
            OUTPUT_POINTS,
            points_list,
            save_lines=False
        )
    else:
        if OUTPUT_LINES is None:
            raise ValueError("OUTPUT_LINES doit être fourni si points_only=False")

        save_CSV(
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
        default=100.0,
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
        INPUT_IMAGE, INPUT_CSV, OUTPUT_POINTS, OUTPUT_LINES = determine_param(image_name)

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

        # construction carte de profondeur
        print("\tConstruction de la carte de profondeur")
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

        if points_only:
            # nouveau dossier pour nuage de points
            output_dir = "../point_cloud/"
            os.makedirs(output_dir, exist_ok=True)
            OUTPUT_POINTS = output_dir + name + ".csv"

            save_CSV(
                OUTPUT_POINTS,
                points_list,
                OUTPUT_LINES=None,
                lines_list=None,
                save_lines=False
            )
        else:
            # comportement normal
            save_CSV(
                OUTPUT_POINTS,
                points_list,
                OUTPUT_LINES,
                lines_list,
                save_lines=True
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
