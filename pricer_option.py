import yfinance as yf
import numpy as np

#Téléchargement des données historiques de l'action
data=yf.download("TTE.PA", period="1y")

#Afficge des 5 dernières lignes des données téléchargées
print(data.tail())

#Calcul des variations quotidiennes en %
data['Rendements journaliers']=data['Close'].pct_change()

#Vérification des rendements calculés
print(data['Rendements journaliers'].tail())

#Calcule de l'écart-type des rendements (Volatilité journalière)
volatilité_quotidienne=data['Rendements journaliers'].std()

#Annualisation des volatilité journalières.
sigma=volatilité_quotidienne*np.sqrt(252) #252 jours de trading par an
print(f"Volatilité annuelle: {sigma:%}")

#Récupération du spot actuel de l'action
S=float(data['Close'].iloc[-1])  #Utilisation de float pour forcer S a être simple.

print(f"Spot actuel de l'action: {S:.2f} EUR")

from scipy.stats import norm

def pricer_option(S,K,T,r,sigma):
    # Calcul du paramètre d1
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

    # Calcul du paramètre d2
    d2=d1-sigma*np.sqrt(T)

    #Calcul du prix du call
    call= S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

    #Calcul du prix du put
    put= K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

    return call, put

#Paramètres de l'option 
K= 60 #Strike 
T= 1 #Maturité en années
r=0.03 #Taux sans risque annuel (OAT 10 ans Française actuellement autour de 3%)

#Appel de la fonction de pricing
prix_call, prix_put= pricer_option(S,K,T,r,sigma)

print(f"Prix théorique du Call Européen: {prix_call:.2f} EUR")
print(f"Prix théorique du Put Européen: {prix_put:.2f} EUR")

import matplotlib.pyplot as plt

#Plage de prix possible du sous-jacent
S_range =np.arange(40,80,1)

#Calcule du prix du call pour chaque scénario
calls = []
for s in S_range:
    calls.append(pricer_option(s, K, T, r, sigma)[0])
#Traçage du graphique
plt.figure(figsize=(10,6))
plt.plot(S_range, calls, label='Prix du Call Européen', color='blue')


#Amélioration du graphique
plt.axvline(x=K, color='red', linestyle='--', label='Strike (K)') #ligne de pointillet rouge sur le strike
plt.grid(True)#Ajout de la grille

#Ajout du titre et les labels des axes
plt.title("Prix du Call Européen en fonction du prix du sous-jacent")
plt.xlabel("Prix du sous-jacent (S)")
plt.ylabel("Prix du Call Européen")

#Ajout de la légende
plt.legend()

#Résultat final
plt.show() 
