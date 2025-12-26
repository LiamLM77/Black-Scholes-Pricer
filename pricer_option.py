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

def monte_carlo_option(S, K, T, r, sigma, n_sims=100000, type_option='call'): # Price d'une option avec une simulation Monte Carlo a l'aide du Mouvement Brownien Géométrique
    np.random.seed(42) # Pour pouvoir reproduire les résultats

    # Tirage aléatoire avec une distribution normale des variables Z
    Z = np.random.standard_normal(n_sims)

    # Simulation du prix à maturité (S_T)
    # Formule du Mouvement Brownien Géométrique
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) 

    # Calcul du payoff à maturité (Gain)
    if type_option == 'call':
        payoffs = np.maximum(S_T - K, 0) # Payoff du call
    else:
        payoffs = np.maximum(K - S_T, 0) # Payoff du put
    
    # Actualisation des payoffs
    prix_mc = np.exp(-r*T) * np.mean(payoffs)
    
    # Calcul de la précision
    # Correction ici : on actualise avec -r*T pour avoir l'écart type du prix présent
    ecart_type = np.std(payoffs * np.exp(-r*T)) 
    erreur_standard = ecart_type / np.sqrt(n_sims) # Erreur standard (loi des grands nombres)

    return prix_mc, erreur_standard

def calcul_delta(S, K, T, r, sigma, type_option='call' ): #Calcul le delta (Nombre d'actions à détenir pour se couvrir)
    if T<=1e-5: #Sécurité si la maturité est atteinte
        if type_option == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) # Calcul du paramètre d1 (Black-Scholes)
    if type_option == 'call':
        delta = norm.cdf(d1)  # Delta du call (entre 0 et 1)
    else:
        delta = norm.cdf(d1) - 1  # Delta du put (entre -1 et 0)
    return delta

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


# Appel de la fonction Monte Carlo
prix_mc, erreur_standard = monte_carlo_option(S, K, T, r, sigma, n_sims=100000)

# Intervalle de confiance à 95% (environ 1.96 fois l'erreur standard)
confiance_95 = 1.96 * erreur_standard

print(f"Prix Monte Carlo : {prix_mc:.4f} EUR")
print(f"Précision (95%)  : +/- {confiance_95:.4f} EUR")
print(f"Intervalle       : [{prix_mc - confiance_95:.4f} ; {prix_mc + confiance_95:.4f}]")

#Test du delta
delta_actuel= calcul_delta(S, K, T, r, sigma, type_option='call')
print(f"Delta actuel du Call: {delta_actuel:.4f}")
print(f"Pour couvrir un call vendu , il faut détenir {delta_actuel:.4f} actions.")

"""
Simule une stratégie de Delta Hedging dynamique.
On vend le Call au début, et on se couvre jour après jour.
"""
def simuler_delta_hedging(S, K, T, r, sigma, nb_steps=252):
    dt= T/nb_steps #Durée d'un pas de temps ( 1 jour de trading)
    dates = np.linspace(0, T, nb_steps + 1) 

    #Génération d'une trajectoire de marché unique ( comme dans la simulation Monte Carlo)
    np.random.seed(42)
    Z = np.random.standard_normal(nb_steps) # Variables aléatoires normales
    S_path = np.zeros(nb_steps + 1) #Matrice pour stocker la trajectoire simulée
    S_path[0] = S #Prix initial

    for i in range(nb_steps):
        S_path[i+1] = S_path[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i]) # Simulation du prix de l'action

    # Initialisation du portefeuille (journée 0)
    prix_call_initial, _ = pricer_option(S, K, T, r, sigma)
    cash = prix_call_initial  # On vend le call et on reçoit le cash

    # On achète delta actions pour se couvrir
    delta= calcul_delta(S, K, T, r, sigma, type_option='call')#Calcul du delta initial
    stock_holdings = delta  # Nombre d'actions détenues
    cash -= delta * S  # Achat des actions avec le cash reçu (on passe en négatif, on emprunte)
    
    #On stocke l'historique pour analyse graphique
    history_deltas = [delta]
    history_cash = [cash]

    #Boucle de Delta Hedging dynamique (Jour par jour du premier au dernier)
    for i in range(1,nb_steps + 1):
        temps_restant = T - dates[i]  # Temps restant jusqu'à maturité

        cash*= np.exp(r*dt)  # Actualisation du cash (intérêts gagnés/perdus)

        if i < nb_steps: #Rebalancement si ce n'est pas le dernier jour
            nouveau_delta= calcul_delta(S_path[i], K, temps_restant, r, sigma, type_option='call')
            transaction= nouveau_delta - stock_holdings # Nombre d'actions à acheter/vendre
            stock_holdings += transaction
            cash -= transaction * S_path[i]  # Ajustement du cash en fonction de la transaction

            history_deltas.append(nouveau_delta)#Stocke le nouveau delta

        else: #Dernier jour, on liquide tout
            
            payoff_final = max(S_path[i] - K, 0)  # Payoff du call à maturité
            
            cash += stock_holdings * S_path[i]  # Vente des actions restantes
            stock_holdings = 0  # Plus d'actions détenues
            
            cash-= payoff_final  # Remboursement du call vendu

            history_deltas.append(0)  # Delta final est 0
            
    return cash, S_path, dates, history_deltas

"""
Visualisation des graphiques

"""
# Lancement de la simulation pour récupérer les données avant d'afficher
pnl_final, S_path, dates, deltas = simuler_delta_hedging(S, K, T, r, sigma)
print(f"PnL Final de la couverture : {pnl_final:.4f} EUR")


plt.figure(figsize=(15, 12)) # Ajusté pour avoir de la place pour 4 graphs

# --- Graphique 1 : Black-Scholes (Le profil de prix) ---

plt.subplot(2, 2, 1) 
S_range = np.arange(40, 80, 1)
calls = [pricer_option(s, K, T, r, sigma)[0] for s in S_range]

plt.plot(S_range, calls, label='Prix du Call', color='navy', linewidth=2)
plt.axvline(x=K, color='red', linestyle='--', label=f'Strike ({K}€)')
plt.grid(True, alpha=0.3) # Grille légère
plt.title("Black-Scholes")
plt.xlabel("Prix de l'action aujourd'hui")
plt.ylabel("Prix de l'option")
plt.legend()

# --- Graphique 2 : Monte Carlo (Les futurs possibles) ---
plt.subplot(2, 2, 2) 

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
plt.title(f"Simulations Monte Carlo ({nb_scenarios} scénarios)")
plt.xlabel("Temps (Années)")
plt.ylabel("Prix de l'action")
plt.legend(loc='upper left')

#--- Graphique 3 : Convergence Monte Carlo vs Black-Scholes ---
plt.subplot(2, 2, 3)

# Paramètres de simulation pour la convergence
N_simulations_visu = 50000  #Nombre total de simulations pour la visualisation

# Simulation de la convergence
np.random.seed(42)
Z = np.random.standard_normal(N_simulations_visu) # Utilise la variable
S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
payoffs = np.maximum(S_T - K, 0)

# Calcul de la moyenne progressive
# Correction importante : np.arange doit aller jusqu'à N + 1 pour avoir la même taille que payoffs
prix_cumules = np.exp(-r*T) * np.cumsum(payoffs) / np.arange(1, N_simulations_visu + 1)

plt.plot(prix_cumules, color='blue', lw=1.5, label='Prix Monte Carlo')
plt.axhline(prix_call, color='red', linestyle='--', lw=2, label=f'Prix Théorique BS ({prix_call:.2f}€)')

plt.title(f"Convergence du prix")
plt.xlabel("Nombre de simulations")
plt.ylabel("Prix estimé du Call")
plt.legend()
plt.grid(True)

#--- Graphique 4 : Delta Hedging ---
plt.subplot(2, 2, 4)

# On affiche l'évolution du Delta (variable 'deltas' récupérée plus haut)
plt.plot(dates, deltas, label='Delta (Actions détenues)', color='orange')
plt.fill_between(dates, deltas, color='orange', alpha=0.3)
plt.title(f"Stratégie Delta Hedging (PnL: {pnl_final:.2f}€)")
plt.xlabel("Temps")
plt.ylabel("Delta")
plt.legend()
plt.grid(True)

plt.tight_layout() 
plt.show()
