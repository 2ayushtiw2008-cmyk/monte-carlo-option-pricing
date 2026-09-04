#imports
import math
import numpy
import scipy.stats
import time
import pandas
import matplotlib.pyplot as plt

# Black Scholes formula


from scipy.stats import norm
def black_scholes(S,K,T,r,sigma):
  d1= (math.log(S/K) + T * (r + (sigma**2)/2)) / (sigma*math.sqrt(T))
  d2= d1 - sigma*math.sqrt(T)
  Nd1= norm.cdf(d1)
  Nd2= norm.cdf(d2)
  call_price= S * Nd1 - K * (math.exp(-r*T)) * Nd2
  return call_price

# Monte Carlo

def monte_carlo(S, K, T, r, sigma, N):
    Z = numpy.random.normal(size=N)
    ST = S * numpy.exp( (r - 0.5 * sigma**2) * T + sigma * numpy.sqrt(T) * Z)
    payoffs = numpy.maximum(ST - K, 0)
    average_payoff = numpy.mean(payoffs)
    # Discount the average payoff back to today
    call_price = numpy.exp(-r * T) * average_payoff
    return call_price

#fixed values
S=100
K=100
T=1
r=0.05
sigma=0.20

#value for black scholes
bs_price = black_scholes(S, K, T, r, sigma)
print("The Black-Scholes price is ", bs_price)

#testing different values of monte carlo with different simulation sizes

simulation_size = [100, 500, 1000, 5000, 10000, 50000, 100000]

runs = 20
results = []
for N in simulation_size:
    errors = []
    start_time = time.perf_counter()
    for i in range(runs):
        mc_price = monte_carlo(S, K, T, r, sigma, N)
        error = abs(mc_price - bs_price)
        errors.append(error)
    end_time = time.perf_counter()
    mean_error = numpy.mean(errors)
    std_error = numpy.std(errors)
    average_runtime = (end_time - start_time) / runs
    results.append([N,bs_price,mean_error,std_error,average_runtime])

# presenting data
results_df = pandas.DataFrame(results, columns = ["Simulations","Black-Scholes Price","Mean Absolute Error","Error Standard Deviation","Average Runtime"])
print(results_df)

plt.figure()
plt.plot(results_df["Simulations"],results_df["Mean Absolute Error"],marker="o")

plt.xlabel("Number of Simulations")
plt.ylabel("Mean Absolute Error")
plt.title("Monte Carlo Pricing Error vs Number of Simulations")
plt.xscale("log")
plt.show()
