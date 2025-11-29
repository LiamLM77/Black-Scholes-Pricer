**Pricer d'Option Black-Scholes & Monte Carlo (Python & Excel)**
Ce projet est un calculateur d'options européennes (Call & Put) basé sur le modèle de Black-Scholes-Merton .

Il compare l'approche analytique classique avec une Simulation de Monte Carlo. 

Il inclut une implémentation Python connectée aux marchés réels et un prototype Excel de validation.

**Objectifs du projet**

Compréhension théorique : Déconstruction des formules mathématiques (d1, d2) via Excel.

Implémentation Python : Automatisation du calcul via scipy.stats.

Données de marché : Récupération automatique du prix Spot et calcul de la volatilité historique via l'API Yahoo Finance (yfinance).

Visualisation : Tracé du profil de prix (Payoff) avec matplotlib.

Approche Numérique : Développement d'un moteur de simulation Monte Carlo (Mouvement Brownien) pour valider la convergence.

**Contenu du Repository**
1. Le Script Python (pricer_option.py)
Le cœur du projet. Il permet de :
Télécharger l'historique d'une action (ex: TotalEnergies).
Calculer la volatilité annualisée.
Pricer un Call et un Put pour un Strike donné.
Pricer un Call et un Put via Monte Carlo (100 000 simulations).
Générer un tableau de bord visuel (Comparaison Théorie vs Simulation).

2. Le Prototype Excel (Pricer_BlackScholes_Excel.xlsx)
Avant l'automatisation, ce fichier a servi à :
Poser les équations manuellement.
Vérifier la cohérence des résultats du script Python (Benchmark).
Visualiser l'impact des taux d'intérêt sur la parité Call-Put.

**Technologies utilisées**
Langage : Python 3.x

Librairies :yfinance : Données financières.

numpy : Calcul matriciel et mathématique.

scipy : Fonctions statistiques (Loi Normale Cumulative).

matplotlib : Dataviz.

Outil de validation : Microsoft Excel

**Installation**
Installer les dépendances :

     pip install yfinance numpy scipy matplotlib

**Lancer le script :**

    python pricer_option.py

**Comment l'utiliser**

Cloner le repo :    
       
       git clone https://github.com/LiamLM77//Option-Pricer-BS-MonteCarlo.git
    
 Ouvrir le dossier dans d'un éditeur de code (VS Code recommandé).
     
Suivre les étapes d'installation ci-dessus.

***Exemple de Résultat (TTE.PA)***
Données de marché :

Volatilité historique (1 an) : ~21.5%

**Comparaison des Prix (Call ATM) :**

| Méthode | Prix Calculé |
| :--- | :--- |
| **Black-Scholes** (Théorie) | **3.90 €** |
| **Monte Carlo** (Simulé) | **3.89 €** |
| *Convergence* | *Validée* |

Visualisation : Le script génère deux graphiques :

Gauche : Le profil de prix théorique (Black-Scholes).

Droite : La projection de 100 trajectoires aléatoires (Spaghetti Plot) montrant la distribution future des prix.

**Disclaimer**
 Projet personnel réalisé en parallèle de mon Master Ingénierie Économique et Financière. À but pédagogique uniquement.

