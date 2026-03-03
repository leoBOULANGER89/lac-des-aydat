#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de triangulation de Delaunay à partir d'un nuage de points CSV.

Ce module permet de générer une surface triangulée au format OBJ à partir d'un nuage de points 3D (x, y, z). La triangulation est
effectuée sur la projection plane (x, y) à l'aide de l'algorithme de Delaunay, tandis que les coordonnées z originales sont conservées
dans le fichier final.

Le script :
    - lit un fichier CSV contenant des points 3D,
    - applique une triangulation de Delaunay en 2D,
    - génère un fichier OBJ contenant les sommets et les faces triangulaires.

Utilisation en ligne de commande
--------------------------------

Traitement d'un fichier CSV (nom sans extension) :

    python script.py --name Lake_Aydat

Optionnel :

    --o chemin/vers/fichier.obj
        Définit le chemin du fichier OBJ de sortie.
        Si non fourni, le fichier est généré automatiquement dans :
            data/final/<name>/<name>_Delaunay.obj

Valeur par défaut :

    --name Lake_Aydat

Logique des options
-------------------

- Le paramètre --name détermine le fichier :
      data/processed/point_cloud/<name>.csv
- Si --o est fourni :
      le fichier OBJ est écrit à l'emplacement spécifié.
- Sinon :
      le dossier de sortie est créé automatiquement et
      le fichier est nommé <name>_Delaunay.obj.

Organisation des fichiers
--------------------------

Entrée :
    - Fichier CSV contenant exactement 3 colonnes : x, y, z

Sortie :
    - Fichier OBJ contenant :
        - les sommets (v x y z)
        - les faces triangulaires (f i j k)

Gestion des erreurs
-------------------

FileNotFoundError
    Si le fichier CSV spécifié n'existe pas.

ValueError
    - Si le CSV ne contient pas exactement 3 colonnes.
    - Si le nombre de points est inférieur à 3.

pandas.errors.EmptyDataError
    Si le fichier CSV est vide.

RuntimeError
    Si la triangulation de Delaunay échoue.

Exception
    Toute erreur inattendue est interceptée et affichée avec son type pour faciliter le débogage.

Notes
-----

- La triangulation est réalisée via scipy.spatial.Delaunay.
- Les dossiers de sortie sont créés automatiquement si nécessaires.
"""


import os
import pandas as pd
from scipy.spatial import Delaunay
from ..io import RAndW

# =============================================================================================================
# PARAMÈTRES
# =============================================================================================================
def determine_param(name, output_override=None):
    """
    Détermine les chemins d'entrée et de sortie pour la triangulation de Delaunay.

    Cette fonction construit :
        - le chemin vers le fichier CSV contenant le nuage de points,
        - le chemin vers le fichier OBJ de sortie contenant le maillage généré.

    Elle crée automatiquement les dossiers de sortie si nécessaire.

    Parameters
    ----------
    name : str
        Nom du jeu de données (sans extension).
        Le fichier d'entrée est supposé se trouver dans : data/processed/point_cloud/<name>.csv
    output_override : str, optional
        Chemin complet personnalisé pour le fichier OBJ de sortie.
        Si fourni, ce chemin est utilisé tel quel.
        Si None (par défaut), le fichier est généré dans data/final/<name>/<name>_Delaunay.obj

    Returns
    -------
    INPUT_CSV : str
        Chemin vers le fichier CSV d'entrée contenant le nuage de points.
    OUTPUT_OBJ : str
        Chemin vers le fichier OBJ de sortie destiné à contenir le maillage triangulé (Delaunay).

    Notes
    -----
    - Les dossiers de sortie sont créés automatiquement via os.makedirs(..., exist_ok=True).
    - Aucun contrôle n'est effectué sur l'existence réelle du fichier CSV.
    - La fonction ne lit ni n'écrit de données ; elle ne fait que déterminer et préparer les chemins.
    """

    data_path = "data/processed/point_cloud/"
    INPUT_CSV = data_path + name + ".csv"

    if output_override is not None:
        OUTPUT_OBJ = output_override
        output_dir = os.path.dirname(OUTPUT_OBJ)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "data/final/" + name + "/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_OBJ = output_dir + name + "_Delaunay.obj"

    return INPUT_CSV, OUTPUT_OBJ


# =============================================================================================================
# DELAUNAY
# =============================================================================================================
def applique_Delaunay(points):
    """
    Applique une triangulation de Delaunay sur un nuage de points 3D.

    La triangulation est effectuée uniquement sur les coordonnées (x, y), c'est-à-dire sur la projection plane du nuage de points.
    Les coordonnées z ne sont pas utilisées dans le calcul de la triangulation.

    Parameters
    ----------
    points : np.ndarray
        Tableau de shape (N, 3) contenant les coordonnées des points.
        Chaque ligne correspond à un point (x, y, z).

    Returns
    -------
    tri : scipy.spatial.Delaunay
        Objet de triangulation retourné par scipy.spatial.Delaunay.
        L'attribut simplices contient les indices des triangles générés.

    Notes
    -----
    - La triangulation est réalisée à l'aide de la classe scipy.spatial.Delaunay.
    - Seules les deux premières colonnes (x, y) sont utilisées.
    - La fonction ne modifie pas le tableau points.
    """

    tri = Delaunay(points[:, :2])
    return tri


# =============================================================================================================
# ÉCRITURE DU OBJ
# =============================================================================================================
def save_obj_Delaunay (OUTPUT_OBJ, tri):
    """
    Sauvegarde une triangulation de Delaunay au format OBJ.

    Cette fonction écrit :
        - les sommets du nuage de points sous la forme "v x y z",
        - les faces triangulaires sous la forme "f i j k".

    Parameters
    ----------
    OUTPUT_OBJ : str
        Chemin du fichier OBJ de sortie.
    tri : scipy.spatial.Delaunay
        Objet de triangulation contenant les triangles dans l'attribut simplices.

    Returns
    -------
    None
        Un fichier .obj est généré à l'emplacement spécifié.

    Notes
    -----
    - Les indices des faces dans le format OBJ commencent à 1, contrairement aux indices Python qui commencent à 0.
      Un décalage de +1 est donc appliqué.
    - Les sommets sont supposés provenir du nuage de points utilisé pour générer la triangulation.
    - Le fichier est ouvert en mode écriture ("w") et sera écrasé s'il existe déjà.
    - La variable points doit être définie dans le scope appelant (non passée explicitement en paramètre).
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
    Fonction principale du pipeline de triangulation de Delaunay.

    Cette fonction :
        1. Lit un nuage de points 3D depuis un fichier CSV,
        2. Applique une triangulation de Delaunay sur la projection (x, y),
        3. Sauvegarde le maillage triangulé au format OBJ.

    Parameters
    ----------
    INPUT_CSV : str
        Chemin vers le fichier CSV contenant le nuage de points.
        Le fichier doit contenir des points 3D sous forme (x, y, z).
    OUTPUT_OBJ : str
        Chemin du fichier OBJ de sortie dans lequel sera écrit
        le maillage triangulé.

    Returns
    -------
    None
        Génère un fichier OBJ contenant :
            - les sommets du nuage de points,
            - les faces issues de la triangulation de Delaunay.

    Notes
    -----
    - La lecture des points est effectuée via la fonction RAndW.read_points.
    - La triangulation est réalisée à l'aide de applique_Delaunay, qui utilise scipy.spatial.Delaunay sur les coordonnées (x, y).
    - L'écriture du fichier OBJ est effectuée par save_obj_Delaunay.
    - La triangulation est calculée en 2D (projection plane), mais le fichier OBJ conserve les coordonnées 3D originales.
    - Le fichier de sortie est écrasé s'il existe déjà.

    Raises
    ------
    FileNotFoundError
        Si le fichier INPUT_CSV n'existe pas.
    ValueError
        Si les données lues ne sont pas au format attendu (par exemple shape incompatible).
    IOError
        En cas d'erreur lors de l'écriture du fichier OBJ.
    """

    points = RAndW.read_points(INPUT_CSV)
    
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
        points = RAndW.read_points(INPUT_CSV)

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