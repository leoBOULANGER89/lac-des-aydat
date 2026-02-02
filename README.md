# lac-des-aydat

## Description du projet
    Projet effectuer dans le cadre de nos études d'ingénieur informatique en 2ème année à l'ISIMA.
    Ce projet consiste à utilisée des méthode vue au cours de notre formation afin de programmée un srcipt en python ou en matlab permettant de pouvoir passée d'un simple nuage de point (en 3D) à une surface correspondante.

### Cas d'utilisation
    Ce type de projet est utile dans différent cas. Par exemple, quand on veux modélisée les fond marain (ou d'un lac), avec un tel programme, il y aurait alors besoin de prendre la profondeur que à certain points.



## 📂 Structure du projet
../
├── donnee/
│ ├── raw/
│ │ └── map/
│ ├── code/
│ └── point_cloud/
├── code/
│ └── V1/
└── resultat/


- **donnee/**  
    Dossier principal contenant l’ensemble des données du projet.

    - **donnee/raw/**  
        Contient les données brutes, non modifiées.

        - **donnee/map/**
            Données sous forme de carte.

            - **donnee/map/x**
                Contien la carte bathymétrique de x. 
                Une version corriger; chaque niveaux (définit par les ligne de niveaux) est colorier d'une couleurs différente.
                Ainsi qu'un ficher txt qui définit la légende.

    - **donnee/code**  
        Scripts utilisés pour le traitement des données brutes. Mes les donnée traité dans donnee/point_cloud/.
    
    - **donnee/point_cloud/**
        Donnée sous forme de nuages de points. Ce format seras celui utilisés dans les différent scripts.

- **code**  
    Dossiée renferment les différentes version des scipts. Ces scripte sont la passsation du nuage de points à une surface 3D.
    - **code/Vxx/**
        Version xx du code pour passée d'un nuage de points à une surface 3D

- **resultat/**  
    Dossiée qui contient les résultats finaux (model 3D, courbe, ...) générés par les scripts.

