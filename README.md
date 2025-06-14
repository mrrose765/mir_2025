# MIR_2025
Moteur de recherche pour le cours de "Multimedia Information Retrieval" de l'Université de Mons.

---

# Installation
Clone le dépôt et se placer dans le répertoire du projet.

```shell
git clone https://github.com/MrRose765/MIR_2025.git && cd MIR_2025
```

## Prérequis
- Docker doit être installé sur votre machine.
- Un serveur X11 doit être en cours d'exécution pour afficher les interfaces graphiques.

## 1 - Construire l'image Docker
Ce docker file est une extension de l'image `coolsa/pyqt-designer:x64` proposé pour le projet en ajoutant la dépendance 
`sentence-transformers` pour la partie multimodale.
````shell
docker build -t mir_project -f Dockerfile .
````

## 2 - Lancer le conteneur Docker
Ne pas oublier de remplacer `<IP_address>` par l'adresse IP correct.  
(Ex: Adresse IP visible dans `MobaXterm`)

````shell
docker run -it --rm -e DISPLAY=<IP_address> -v "$(pwd):/opt/project" mir_project
````

## 3 - Télécharger les features et données
Dans le conteneur Docker, lancer le script de téléchargement des données et des features
```shell
./download_data.sh
```

## 4 - Lancer l'application
Dans le conteneur Docker, naviguer vers le répertoire `/opt/project/src` avec
```shell
cd /opt/project/src
```
et lancer l'application:
````shell
python3 main_app.py
````
