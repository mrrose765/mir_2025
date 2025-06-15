# MIR_2025
Moteur de recherche pour le cours de "Multimedia Information Retrieval" de l'Université de Mons.

---

# Installation
Clone le dépôt et se placer dans le répertoire du projet.

```shell
git clone --depth=1 https://github.com/MrRose765/MIR_2025.git && cd MIR_2025
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

# Notes
Si, lors de l'installation, vous rencontrez une erreur comme :
```shell
bash: ./download_data.sh: /bin/bash^M: bad interpreter: No such file or directory
```
Cela signifie que le fichier `download_data.sh` a été créé ou modifié automatiquement sur votre machine, 
ce qui peut introduire des caractères de fin de ligne incompatibles avec Linux. ([source](https://stackoverflow.com/questions/14219092/bash-script-bin-bashm-bad-interpreter-no-such-file-or-directory))

Pour corriger cela, vous pouvez utiliser la commande suivante sur le conteneur Docker pour convertir le fichier en format Unix :

```shell
sed -i 's/\r$//' download_data.sh
```

ou

```shell
apt-get install -y dos2unix
dos2unix download_data.sh
```
