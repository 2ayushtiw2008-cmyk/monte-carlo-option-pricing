# Applying it to NVIDIA

#imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S / K)+ T * (r + (sigma**2) / 2)) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    call_price = (S * Nd1 - K * math.exp(-r * T) * Nd2)
    return call_price

def monte_carlo(S, K, T, r, sigma, N):
    Z = np.random.normal(size=N)
    ST = S * np.exp((r - 0.5 * sigma**2) * T+ sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    average_payoff = np.mean(payoffs)
    call_price = np.exp(-r * T) * average_payoff
    return call_price

# downloading NVIDIA's data
ticker = "NVDA"
data = yf.download(ticker,period="3y",interval="1d",auto_adjust=False)
print("First few rows:")
print(data.head())

print("\nLast few rows:")
print(data.tail())

# adjusted closing prices

prices = data["Adj Close"].squeeze().dropna()
print("\nNumber of observations:", len(prices))
print("First date:", prices.index[0])
print("Last date:", prices.index[-1])


#daily log returns and historical volatility

log_returns = np.log(prices / prices.shift(1)).dropna()
print("\nFirst few daily log returns:")
print(log_returns.head())


daily_volatility = log_returns.std()

annualised_volatility = (daily_volatility * np.sqrt(252))
print("\nDaily volatility:",round(daily_volatility, 6))

print("Annualised volatility:",round(annualised_volatility, 4))

print("Annualised volatility (%):",round(annualised_volatility * 100, 2), "%")

#Graph 1 - NVIDIA Share Price

plt.figure()
plt.plot(prices)
plt.xlabel("Date")
plt.ylabel("Share Price ($)")
plt.title("NVIDIA Share Price - Last 3 Years")

plt.show()

#Graph 2: NVIDIA's daily log returns

plt.figure()
plt.plot(log_returns)
plt.xlabel("Date")
plt.ylabel("Daily Log Return")
plt.title("NVIDIA Daily Log Returns")

plt.show()


# Latest NVIDIA adjusted closing price
S = float(prices.iloc[-1])

# At-the-money strike price
K = S

# One year until expiry
T = 1

# Approximate 1-year US risk-free rate
r = 0.041

# Historical NVIDIA volatility

sigma = annualised_volatility

# Black-Scholes


bs_price = black_scholes(S,K,T,r,sigma)

#Monte Carlo

N = 100000

mc_prices = []
for i in range(20):
    mc_price = monte_carlo(S,K,T,r,sigma,N)
    mc_prices.append(mc_price)

average_mc_price = np.mean(mc_prices)

errors = []

for price in mc_prices:
    error = abs(price - bs_price)
    errors.append(error)

mean_error = np.mean(errors)

percentage_error = (mean_error / bs_price) * 100

# printing results

print("\n========================================")
print("NVIDIA EUROPEAN CALL OPTION")
print("========================================")

print("Share price: $",round(S, 2))

print("Strike price: $",round(K, 2))

print("Historical volatility:",round(sigma * 100, 2), "%")

print("Time to maturity:",T, "year")

print("Risk-free rate:",r * 100, "%")

print("Monte Carlo simulations:",N)

print("\nBlack-Scholes price: $",round(bs_price, 4))

print("Average Monte Carlo price: $",round(average_mc_price, 4))

print("Mean absolute error: $",
      round(mean_error, 4))

print("Mean percentage error:",round(percentage_error, 4), "%")

#Black-Scholes Vs Monte Carlo

model_names = ["Black-Scholes","Monte Carlo"]

model_prices = [bs_price,average_mc_price]

plt.figure()
plt.bar(model_names,model_prices)
plt.ylabel("Option Price ($)")
plt.title("NVIDIA Call Option Price")

plt.show()

# Monte-Carlo Prices

plt.figure()
plt.plot(range(1, 21),mc_prices,marker="o")

plt.axhline(bs_price,linestyle="--",label="Black-Scholes Price")

plt.xlabel("Monte Carlo Run")
plt.ylabel("Option Price ($)")
plt.title("Monte Carlo Estimates vs Black-Scholes")
plt.legend()

plt.show()
