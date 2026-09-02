# Black-Scholes option pricing model
import math
from scipy.stats import norm
def black_scholes(S,K,T,r,sigma):
  d1=(math.log(S/K)+T*(r+(sigma**2)/2))/(sigma*math.sqrt(T))
  d2=d1-sigma*math.sqrt(T)
  Nd1= norm.cdf(d1)
  Nd2= norm.cdf(d2)
  call_price=S*Nd1-K*(math.exp(-r*T))*Nd2
  return call_price
price=black_scholes(100, 100, 1, 0.05, 0.2)
print(price)
