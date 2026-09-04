# imports
import math
import numpy
import scipy.stats
import pandas
import matplotlib.pyplot as plt

# Black Scholes formula

from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):

    d1 = (math.log(S / K) + T * (r + (sigma**2) / 2)) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    call_price = S * Nd1 - K * math.exp(-r * T) * Nd2

    return call_price


# Monte Carlo

def monte_carlo(S, K, T, r, sigma, N):

    Z = numpy.random.normal(size=N)

    ST = S * numpy.exp((r - 0.5 * sigma**2) * T+ sigma * numpy.sqrt(T) * Z)

    payoffs = numpy.maximum(ST - K, 0)

    average_payoff = numpy.mean(payoffs)

    # Discounting
    call_price = numpy.exp(-r * T) * average_payoff

    return call_price


# fixed values

S = 100
K = 100
T = 1
r = 0.05
N = 10000


volatility_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
runs = 20
results = []


# Testing volatilitiies

for sigma in volatility_values:

    bs_price = black_scholes(S, K, T, r, sigma)

    mc_prices = []
    errors = []
    for i in range(runs):

        mc_price = monte_carlo(S, K, T, r, sigma, N)
        mc_prices.append(mc_price)
        error = abs(mc_price - bs_price)

        errors.append(error)
    average_mc_price = numpy.mean(mc_prices)
  
    mean_error = numpy.mean(errors)

    std_error = numpy.std(errors)

    results.append([sigma,bs_price,average_mc_price,mean_error,std_error])


# Presenting data

results_df = pandas.DataFrame(results,columns=["Volatility","Black-Scholes Price","Average Monte Carlo Price","Mean Absolute Error","Error Standard Deviation"])
print(results_df)


# Graph 1: Volatility vs Option Price

plt.figure()

plt.plot(results_df["Volatility"],results_df["Black-Scholes Price"],marker="o",label="Black-Scholes")

plt.plot(results_df["Volatility"],results_df["Average Monte Carlo Price"],marker="o",label="Monte Carlo")

plt.xlabel("Volatility")
plt.ylabel("Option Price")

plt.title("Effect of Volatility on European Call Option Price")

plt.legend()

plt.show()


# Graph 2: Volatility vs Monte Carlo Error

plt.figure()

plt.plot(results_df["Volatility"],results_df["Mean Absolute Error"],marker="o")

plt.xlabel("Volatility")
plt.ylabel("Mean Absolute Error")

plt.title("Effect of Volatility on Monte Carlo Pricing Error")

plt.show()
