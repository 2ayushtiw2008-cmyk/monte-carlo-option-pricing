import numpy

def monte_carlo(S, K, T, r, sigma, N):
    Z = numpy.random.normal(size=N)
    ST = S * numpy.exp( (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = numpy.maximum(ST - K, 0)
    average_payoff = numpy.mean(payoffs)
    # Discount the average payoff back to today
    call_price = numpy.exp(-r * T) * average_payoff
    return call_price
