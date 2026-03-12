# Reconstruction 3D d'une surface avec une carte des courbe de niveaux.

## Description du projet
    Projet effectuer dans le cadre de nos études d'ingénieur informatique en 2ème année à l'ISIMA.
    Ce projet vise à reconstruire une **surface 3D** à partir d'une **carte bathymétrique**.  
Le pipeline permet de transformer une **image contenant des courbes de niveau** en **maillage 3D triangulé** exploitable en simulation.
    **Nous avont appliquer ce processus afin au cas pratique du lac d'aydat**

---

## Pipeline
Le processus comprend plusieurs étapes :

1. Extraction des profondeurs depuis l'image
2. Reconstruction des courbes de niveau
3. Génération d'un nuage de points
4. Triangulation de Delaunay contrainte
5. Export du maillage au format `.obj`

---

## Structure du projet
../  
├── data/  
│ ├── final/  
| | ├── Lake_Aydat_Bicubic_Delaunay.obj  
| | ├── Lake_Aydat_CDT.obj  
| | ├── Lake_Aydat_curves.png  
| | ├── Lake_Aydat_Delaunay.obj  
| | ├── Lake_Aydat_mesure_simu.png  
| | └── Lake_Aydat_nuage_points.png  
| |  
│ ├── processed/  
│ | ├── curves/  
│ | | ├── Lake_Aydat_lines.csv  
│ | | └── Lake_Aydat_points.csv  
│ | └── point_cloud/  
│ |   └── Lake_Aydat.csv  
| |
│ └── raw/  
|   └── map/  
|     └── Lake_Aydat/  
|       ├── Lake_Aydat_traitee.png  
|       └── légende.csv  
|  
└── src/  
  ├── extraction/  
  │ └── traitement.py  
  ├── io/  
  │ └── RAndW.py  
  ├── mesure/  
  │ └── mesure.py  
  ├── processing/  
  │ ├── Bicubic.py  
  │ ├── CDT.py  
  │ └── Delaunay.py  
  └── visualisation/  
    └── visualisee.py  


- **data/**  
  Dossier principal contenant l’ensemble des données du projet.  
    - **data/raw/**  
      Contient les données brutes, non modifiées par les scriptes.  
        - **data/raw/map/x**  
          Contien la carte bathymétrique de x.   
          Une version corriger de cet carte; chaque niveaux (définit par les ligne de niveaux) est colorier d'une couleurs différente (on recommende de le faire avec un niveaux de gris) sans ligne de separation de niveaux.  
          Ainsi qu'un ficher légende.txt qui définit fait correspondre a chaque couleurs, le niveaux le plus bas possible..  
    - **data/processed/**  
      Dossier contenant l’ensemble des données qui ont deja ete traiter.  
        - **data/processed/point_cloud**  
          Dossier contenant l’ensemble des cartes de niveaux transformer en nuage de point par **src/extraction/traitement.py** sous forme de **x.csv**.  
        - **data/processed/curves**  
          Dossier contenant l’ensemble des cartes de niveaux approximer par des droites via **src/extraction/traitement.py**.  
          **x__lines.csv** contien les droites d'approximation.  
          **x_points.csv** contien les points servant a l'approximation.  
    - **data/final/x**  
      Dossier contenant l’ensemble des resultats de x que ce soit les courbes de mesure fait via **src/mesure/mesure.py** ou toutes les surfaces final fait a partir des scriptes de **src/processing/**  

- **src**  
  Dossiée renferment les scriptes utiliser.  
    - **src/extraction/**  
      Contien les scripte pour extraire les information des cartes.
        - **src/extraction/traitement.py**  

