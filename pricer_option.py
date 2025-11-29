import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
Sélection de l'action , calcule de la volatilité anualiée (sigma) et récupération du spot actuel (S)
"""

#Téléchargement des données historiques de l'action
data = yf.download("TTE.PA", period="1y", progress=False)

#Afficge des 5 dernières lignes des données téléchargées
print(data.tail())

#Calcul des variations quotidiennes en %
data['Rendements journaliers'] = data['Close'].pct_change()

#Vérification des rendements calculés
print(data['Rendements journaliers'].tail())

#Calcule de l'écart-type des rendements (Volatilité journalière)
volatilité_quotidienne = data['Rendements journaliers'].std()

#Annualisation des volatilité journalières.
sigma = volatilité_quotidienne * np.sqrt(252) #252 jours de trading par an
print(f"Volatilité annuelle: {sigma:%}")

#Récupération du spot actuel de l'action
S = float(data['Close'].iloc[-1])  #Utilisation de float pour forcer S a être simple.

print(f"Spot actuel de l'action: {S:.2f} EUR")

"""
Fonctions utilisées. Black-Scholes pour le benchmark / Monte Carlo pour la simulation.
"""

def pricer_option(S, K, T, r, sigma):
    # Calcul du paramètre d1
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Calcul du paramètre d2
    d2 = d1 - sigma * np.sqrt(T)

    #Calcul du prix du call
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    #Calcul du prix du put
    put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call, put

def monte_carlo_option(S, K, T, r, sigma, n_sims=100000, type_option='call'): #Price d'une option avec une simulation Monte Carlo a l'aide du Mouvement Brownien Géométrique
    np.random.seed(42) #Pour pouvoir reproduire les résultats

    #Tirage aléatoire avec une distribution normale des variables Z
    Z = np.random.standard_normal(n_sims)

    #Simulation du prix à maturité (S_T)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  #Formule du Mouvement Brownien Géométrique

    #Calcul du payoff à maturité (Gain)
    if type_option == 'call':
        payoffs = np.maximum(S_T - K, 0) #Payoff du call
    else:
        payoffs = np.maximum(K - S_T, 0) #Payoff du put
    #Actualisation des payoffs
    prix = np.exp(-r*T) * np.mean(payoffs)
    return prix

"""
Exécution des calculs
"""

#Paramètres de l'option 
K = 60 #Strike 
T = 1 #Maturité en années
r = 0.03 #Taux sans risque annuel (OAT 10 ans Française actuellement autour de 3%)

#Appel de la fonction de pricing théorique (Black-Scholes)
prix_call, prix_put = pricer_option(S, K, T, r, sigma)

print(f"Prix théorique du Call Européen: {prix_call:.2f} EUR")
print(f"Prix théorique du Put Européen: {prix_put:.2f} EUR")

#Appel de la fonction Monte Carlo (Pour comparer)
prix_call_mc = monte_carlo_option(S, K, T, r, sigma, type_option='call')
print(f"Prix Monte Carlo du Call: {prix_call_mc:.2f} EUR")


"""
Visualisation des graphiques

"""
plt.figure(figsize=(15, 6))

# --- Graphique 1 : Black-Scholes (Le profil de prix) ---

plt.subplot(1, 2, 1) 
S_range = np.arange(40, 80, 1)
calls = [pricer_option(s, K, T, r, sigma)[0] for s in S_range]

plt.plot(S_range, calls, label='Prix du Call', color='navy', linewidth=2)
plt.axvline(x=K, color='red', linestyle='--', label=f'Strike ({K}€)')
plt.grid(True, alpha=0.3) # Grille légère
plt.title("Black-Scholes)")
plt.xlabel("Prix de l'action aujourd'hui")
plt.ylabel("Prix de l'option")
plt.legend()

# --- Graphique 2 : Monte Carlo (Les futurs possibles) ---
plt.subplot(1, 2, 2) 

# Paramètres de simulation pour le graph
nb_scenarios = 100 #Nombre de trajectoires simulées
nb_steps = 252 #Nombre de pas de temps (jours de trading dans une année)
dt = T / nb_steps
t_axis = np.linspace(0, T, nb_steps + 1) 
S_paths = np.zeros((nb_steps + 1, nb_scenarios)) #Matrice pour stocker les trajectoires simulées
S_paths[0] = S  #Prix initial

# Simulation
for i in range(1, nb_steps + 1):
    Z = np.random.standard_normal(nb_scenarios)
    S_paths[i] = S_paths[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z) 

# Amélioration de la visualisation:
# Traçage de toutes les trajectoires en gris très fin et transparent
plt.plot(t_axis, S_paths, color='grey', alpha=0.4, linewidth=0.5)

#Calcule et traçage de la moyenne des trajectoires (en bleu plus épais)
S_average = np.mean(S_paths, axis=1)
plt.plot(t_axis, S_average, color='blue', linewidth=3, label='Moyenne des scénarios')

# La ligne du Strike
plt.axhline(K, color='red', linestyle='--', linewidth=2, label=f'Strike ({K}€)')

plt.grid(True, alpha=0.3)
plt.title(f"2. Simulations Monte Carlo du prix de l'action sur {T} an avec {nb_scenarios} scénarios")
plt.xlabel("Temps (Années)")
plt.ylabel("Prix de l'action")
plt.legend(loc='upper left')

plt.tight_layout() 
plt.show()
