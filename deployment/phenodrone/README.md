# Installation multiutilisateur sur phenodrone

Cette procédure installe une unique copie partagée de PhenoLapseUAV dans `/opt/phenodrone` et ajoute le lanceur **PhenoLapseUAV** au menu des applications Linux. Les utilisateurs peuvent lancer l'application, mais seuls les administrateurs modifient le dépôt et l'environnement.

## Prérequis administrateur

- Une session graphique Linux par utilisateur (écran local, NoMachine ou VNC de préférence).
- Git et Python 3.11 ou plus récent installés sur le serveur.
- Un accès réseau au dépôt Git et à PyPI lors de l'installation et des mises à jour.
- Droits `sudo` pour l'administrateur chargé de la maintenance.

Pour une utilisation par redirection X11, lancer une session avec `ssh -X` (ou `ssh -Y` si la politique locale le permet) puis démarrer `phenolapseuav`. L'interface reste fonctionnelle, mais l'affichage de grandes orthomosaïques peut être sensiblement plus lent qu'avec NoMachine/VNC ou un écran local.

## Première installation

Depuis un clone administrateur de ce dépôt, exécuter une seule fois le script avec l'URL Git officielle et la branche à déployer :

`sudo ./deployment/phenodrone/install.sh https://github.com/Rocsg/phenolapseUAV.git main`

Pour employer un Python différent, fournir `PYTHON_BIN` au lancement, par exemple `PYTHON_BIN=python3.12`.

Le script :

1. clone ou actualise le dépôt central dans `/opt/phenodrone/phenolapseUAV` ;
2. crée `/opt/phenodrone/venv` ;
3. installe napari avec le backend graphique PyQt5, puis le paquet et ses dépendances dans cet environnement ;
4. crée les commandes communes `/opt/phenodrone/bin/phenolapseuav` et `/opt/phenodrone/bin/phenolapseuav-update` ;
5. ajoute le lanceur graphique système.

## Utilisation

Chaque utilisateur démarre **PhenoLapseUAV** depuis le menu des applications ou exécute `/opt/phenodrone/bin/phenolapseuav` depuis un terminal graphique. Le lanceur ouvre napari depuis l'environnement partagé ; napari découvre ensuite le plugin PhenoLapseUAV dans son menu **Plugins**.

Les images, états de recalage et exports restent dans les dossiers de données des utilisateurs. Ne pas donner de droits d'écriture ordinaires sur `/opt/phenodrone`.

## Mise à jour Git

Le choix assumé ici est de toujours déployer le dernier commit de la branche configurée, sans conserver de répertoires de versions. L'administrateur doit prévenir les utilisateurs et attendre qu'ils aient fermé PhenoLapseUAV, puis exécuter :

`sudo /opt/phenodrone/bin/phenolapseuav-update`

Cette commande effectue uniquement un `git pull --ff-only`, réinstalle le paquet dans l'environnement commun et vérifie les dépendances. Elle refuse d'écraser des changements locaux non validés dans le clone central.

Sans archive de versions, une mise à jour défectueuse doit être annulée par Git avec le commit précédent, puis suivie d'une nouvelle exécution de la commande de mise à jour. Tester les évolutions sur une autre machine avant de mettre à jour phenodrone reste donc important.
