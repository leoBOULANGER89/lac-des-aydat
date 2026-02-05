#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génération d'une surface triangulée par Delaunay à partir d'un nuage de points 3D.

Ce programme permet de :
- lire un nuage de points 3D depuis un fichier CSV,
- appliquer une triangulation de Delaunay sur les coordonnées (x, y),
- générer une surface triangulaire,
- exporter cette surface au format OBJ.

Le fichier OBJ généré contient :
- les sommets 3D (v x y z),
- les faces triangulaires (f i j k), compatibles avec les logiciels 3D
  tels que Blender, MeshLab ou CloudCompare.

Le fichier de sortie peut être généré automatiquement ou spécifié
explicitement par l'utilisateur.

Entrées attendues
-----------------
Nuage de points :
    CSV contenant exactement les colonnes :
        - x : coordonnée X
        - y : coordonnée Y
        - z : coordonnée Z

Sorties générées
----------------
Surface triangulée :
    Fichier OBJ contenant la triangulation de Delaunay.

Par défaut :
    ../resultat/<name>/<name>_Delaunay.obj

Utilisation
-----------
python delaunay.py
python delaunay.py --name Lake_Aydat
python delaunay.py --name Lake_Aydat --o ./surface.obj

Options
-------
--name
    Nom du fichier CSV à traiter (sans extension).
--o, --output
    Chemin du fichier OBJ de sortie.

Notes
-----
- La triangulation de Delaunay est effectuée uniquement dans le plan (x, y).
- Les coordonnées z sont conservées pour la géométrie 3D finale.
- Un minimum de 3 points est requis pour appliquer l'algorithme.
- Les dossiers de sortie sont créés automatiquement s'ils n'existent pas.
"""


import os
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# =============================================================================================================
# PARAMÈTRES
# =============================================================================================================
def determine_param(name, output_override=None):
    """
    Détermine les chemins des fichiers d'entrée (CSV) et de sortie (OBJ).

    Le fichier d'entrée est construit à partir du nom du jeu de données.
    Le fichier de sortie peut soit être généré automatiquement, soit être
    explicitement fourni via un chemin personnalisé.

    Parameters
    ----------
    name : str
        Nom du fichier CSV à traiter (sans l'extension `.csv`).
    output_override : str or None, optional
        Chemin du fichier OBJ de sortie.
        Si None, le fichier OBJ est généré automatiquement dans
        `../resultat/<name>/<name>_Delaunay.obj`.

    Returns
    -------
    INPUT_CSV : str
        Chemin vers le fichier CSV contenant le nuage de points.
    OUTPUT_OBJ : str
        Chemin vers le fichier OBJ de sortie.

    Notes
    -----
    - Les répertoires nécessaires à l'écriture du fichier OBJ sont créés
      automatiquement s'ils n'existent pas.
    - Le fichier d'entrée CSV n'est jamais modifié par ce paramétrage.
    """

    data_path = "../donnee/point_cloud/"
    INPUT_CSV = data_path + name + ".csv"

    if output_override is not None:
        OUTPUT_OBJ = output_override
        output_dir = os.path.dirname(OUTPUT_OBJ)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "../resultat/" + name + "/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_OBJ = output_dir + name + "_Delaunay.obj"

    return INPUT_CSV, OUTPUT_OBJ


# =============================================================================================================
# LECTURE DU CSV
# =============================================================================================================
def extraction_points (INPUT_CSV):
    """
    Lit un fichier CSV contenant des coordonnées 3D et extrait
    les points sous forme de tableau numpy.

    Le fichier CSV doit contenir les colonnes : x, y, z.

    Parameters
    ----------
    INPUT_CSV : str
        Chemin vers le fichier CSV.

    Returns
    -------
    points : numpy.ndarray
        Tableau de forme (N, 3) contenant les coordonnées (x, y, z).
    """

    data = pd.read_csv(INPUT_CSV)
    points = data[["x", "y", "z"]].values
    return points


# =============================================================================================================
# DELAUNAY
# =============================================================================================================
def applique_Delaunay(points):
    """
    Applique une triangulation de Delaunay sur un ensemble de points 3D.

    La triangulation est effectuée uniquement sur les coordonnées (x, y).

    Parameters
    ----------
    points : numpy.ndarray
        Tableau de points 3D de forme (N, 3).

    Returns
    -------
    tri : scipy.spatial.Delaunay
        Objet Delaunay contenant les triangles (simplices).
    """
        
    tri = Delaunay(points[:, :2])
    return tri


# =============================================================================================================
# ÉCRITURE DU OBJ
# =============================================================================================================
def save_obj_Delaunay (OUTPUT_OBJ, tri):
    """
    Sauvegarde une triangulation de Delaunay au format OBJ.

    Le fichier OBJ contient :
    - les sommets (v x y z)
    - les faces triangulaires (f i j k)

    Parameters
    ----------
    OUTPUT_OBJ : str
        Chemin du fichier OBJ de sortie.
    tri : scipy.spatial.Delaunay
        Triangulation de Delaunay à exporter.
    """

    with open(OUTPUT_OBJ, "w") as f:
        # Sommets
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        # Faces (indices OBJ commencent à 1)
        for simplex in tri.simplices:
            i, j, k = simplex + 1
            f.write(f"f {i} {j} {k}\n")



# ======================
# MAIN
# ======================
def Delaunay_main(INPUT_CSV, OUTPUT_OBJ):
    """
    Fonction principale appliquant une triangulation de Delaunay
    à un nuage de points issu d'un fichier CSV.

    Parameters
    ----------
    name : str
        Nom du fichier CSV (sans extension).
    
    Returns
    -------
    None
    """
    points = extraction_points(INPUT_CSV)
    
    #application de Delaunay
    tri = applique_Delaunay(points)

    #Sauvegarder le model 3D
    save_obj_Delaunay (OUTPUT_OBJ, tri)



if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description="Application d'une triangulation de Delaunay à partir d'un nuage de points CSV."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Lake_Aydat",
        help="Nom du fichier CSV à traiter (sans l'extension .csv). "
             "Par défaut : Lake_Aydat"
    )

    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier OBJ de sortie. "
             "Si non fourni, le fichier est généré automatiquement."
    )

    args = parser.parse_args()
    name = args.name
    output_override = args.o

    print("Utilisation de Delaunay :")

    try:
        # Détermination des fichiers
        print(f"\t📁 Récupération des fichiers pour {name}")
        INPUT_CSV, OUTPUT_OBJ = determine_param(name, output_override)

        if not os.path.isfile(INPUT_CSV):
            raise FileNotFoundError(f"Fichier CSV introuvable : {INPUT_CSV}")

        # Lecture des points
        print(f"\t📊 Extraction des points depuis {INPUT_CSV}")
        points = extraction_points(INPUT_CSV)

        if points.shape[1] != 3:
            raise ValueError("Le fichier CSV doit contenir exactement 3 colonnes : x, y, z")

        if points.shape[0] < 3:
            raise ValueError("Au moins 3 points sont nécessaires pour appliquer Delaunay")

        # Delaunay
        print("\t🔺 Application de Delaunay")
        try:
            tri = applique_Delaunay(points)
        except Exception as e:
            raise RuntimeError(f"Échec de la triangulation de Delaunay : {e}")

        # Sauvegarde
        print("\t💾 Sauvegarde de l'objet OBJ")
        save_obj_Delaunay(OUTPUT_OBJ, tri)

        print(f"\t✅ Surface par Delaunay enregistrée : {OUTPUT_OBJ}")

    except FileNotFoundError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except pd.errors.EmptyDataError:
        print("\n❌ ERREUR : Le fichier CSV est vide")
        sys.exit(1)

    except KeyError:
        print("\n❌ ERREUR : Le CSV doit contenir les colonnes x, y, z")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except RuntimeError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except Exception as e:
        print("\n❌ ERREUR INATTENDUE")
        print(f"   {type(e).__name__} : {e}")
        sys.exit(1)