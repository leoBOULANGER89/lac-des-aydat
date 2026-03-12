#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de génération d'une surface 3D via Poisson.

Ce module permet de reconstruire une surface 3D à partir d'un nuage de points (fichier CSV) en utilisant une triangulation Poisson, 
puis de sauvegarder le maillage au format OBJ.

Utilisation en ligne de commande
--------------------------------

Traitement d'un jeu de données par défaut (Lake_Aydat) :

    python script.py

Spécifier un fichier CSV à traiter :

    python script.py --name chemin/vers/points.csv

Optionnel :

    --o chemin/fichier.obj
        Définit le chemin du fichier OBJ de sortie.
        Par défaut : data/final/{name}/{name}_Poisson.obj

    --alpha valeur_float
        Paramètre alpha contrôlant la taille maximale des triangles dans la triangulation 2D (par défaut 1000.0).

Logique des options
-------------------

- Si --name n'est pas fourni :
       le jeu de données par défaut "Lake_Aydat" est utilisé.
- Si --o n'est pas fourni :
       le fichier de sortie est généré automatiquement dans data/final/{name}/{name}_Poisson.obj.
- Si --alpha est fourni :
       contrôle le raffinement de la triangulation 2D dans le plan XY.

Organisation des fichiers
--------------------------

Entrée :
    - Fichier CSV contenant le nuage de points (x, y, z).

Sortie :
    - Fichier OBJ contenant la surface reconstruite par Poisson.

Les dossiers de sortie sont créés automatiquement si nécessaires.

Gestion des erreurs
-------------------

FileNotFoundError
    Si le fichier CSV spécifié n'existe pas.

pd.errors.EmptyDataError
    Si le fichier CSV est vide.

KeyError
    Si le CSV ne contient pas les colonnes attendues.

ValueError
    Si le CSV contient moins de 3 points ou un nombre de colonnes incorrect.

RuntimeError
    Si la reconstruction Poisson échoue pour une raison interne.

Exception
    Pour toute autre erreur inattendue, un message détaillé est affiché.
"""

import os
import sys
import pandas as pd
import pyvista as pv

from ..io import RAndW


# =============================================================================================================
# PARAMÈTRES
# =============================================================================================================
def determine_param(name, output_override=None):
    """
    Détermine les chemins d'entrée et de sortie pour la reconstruction d'une surface 3D via Poisson Surface Reconstruction.

    Parameters
    ----------
    name : str
        Nom du jeu de données (sans extension). Utilisé pour construire automatiquement le chemin vers le fichier CSV d'entrée :
            data/processed/point_cloud/{name}.csv
    output_override : str or None, optional
        Chemin complet du fichier OBJ de sortie.
        Si fourni, ce chemin est utilisé directement et son dossier parent est créé si nécessaire.
        Si None, le fichier de sortie est généré automatiquement dans :
            data/final/{name}/{name}_Poisson.obj

    Returns
    -------
    INPUT_CSV : str
        Chemin vers le fichier CSV contenant le nuage de points.
    OUTPUT_OBJ : str
        Chemin vers le fichier OBJ de sortie pour la surface reconstruite.

    Notes
    -----
    - Les dossiers de sortie sont créés automatiquement s'ils n'existent pas.
    - Cette fonction centralise la logique des chemins pour éviter la duplication dans le reste du programme.
    """

    INPUT_CSV = f"data/processed/point_cloud/{name}.csv"

    if output_override is not None:
        OUTPUT_OBJ = output_override
        output_dir = os.path.dirname(OUTPUT_OBJ)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"data/final/{name}/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_OBJ = os.path.join(output_dir, f"{name}_Poisson.obj")

    return INPUT_CSV, OUTPUT_OBJ



# =============================================================================================================
# POISSON-LIKE (TRIANGULATION 2D)
# =============================================================================================================
def applique_poisson(points, alpha=1000.0):
    """
    Applique une reconstruction de surface via Poisson à partir d'un nuage de points 3D.

    Parameters
    ----------
    points : np.ndarray
        Tableau de shape (N, 3) contenant les coordonnées x, y, z des points du nuage.
    alpha : float, optional
        Paramètre de lissage pour la triangulation 2D dans le plan XY.
        Valeur par défaut : 1000.0. Plus la valeur est grande, plus la triangulation respecte finement les points du nuage.

    Returns
    -------
    mesh : pyvista.PolyData
        Objet PolyData contenant la surface reconstruite avec :
            - triangles formant la surface,
            - normales calculées sur les cellules (cell_normals=True).

    Notes
    -----
    - La triangulation est réalisée dans le plan XY.
    - Les normales sont calculées pour chaque cellule et non pour les points.
    - L'objet retourné peut être directement utilisé pour l'affichage ou l'export au format OBJ via mesh.save(filename).
    """

    cloud = pv.PolyData(points)

    # Triangulation 2D dans le plan XY
    mesh = cloud.delaunay_2d(alpha=alpha)

    # Calcul des normales de cellules
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)

    return mesh


# =============================================================================================================
# SAUVEGARDE OBJ
# =============================================================================================================
def save_obj_poisson(output_obj, mesh):

    mesh.save(output_obj)


# =============================================================================================================
# MAIN
# =============================================================================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Génération d'une surface Poisson via PyVista."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Lake_Aydat",
        help="Nom du fichier CSV à traiter (sans extension)."
    )

    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier OBJ de sortie."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1000.0,
        help="Paramètre alpha contrôlant la taille maximale des triangles."
    )

    args = parser.parse_args()

    print("Utilisation de Poisson :")

    try:
        INPUT_CSV, OUTPUT_OBJ = determine_param(args.name, args.o)

        if not os.path.isfile(INPUT_CSV):
            raise FileNotFoundError(f"Fichier CSV introuvable : {INPUT_CSV}")

        print("\t📊 Lecture des points")
        points = RAndW.read_points(INPUT_CSV)

        if points.shape[0] < 3:
            raise ValueError("Au moins 3 points sont nécessaires")

        if points.shape[1] != 3:
            raise ValueError("Le CSV doit contenir exactement 3 colonnes : x, y, z")

        print("\t🔺 Génération surface Poisson")
        try:
            mesh = applique_poisson(points, alpha=args.alpha)
        except Exception as e:
            raise RuntimeError(f"Échec de la reconstruction : {e}")

        print("\t💾 Sauvegarde OBJ")
        save_obj_poisson(OUTPUT_OBJ, mesh)

        print(f"\t✅ Surface Poisson enregistrée : {OUTPUT_OBJ}")

    except FileNotFoundError as e:
        print(f"\n❌ ERREUR : {e}")
        sys.exit(1)

    except pd.errors.EmptyDataError:
        print("\n❌ ERREUR : Le fichier CSV est vide")
        sys.exit(1)

    except KeyError as e:
        print(f"\n❌ ERREUR : {e}")
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