from scipy.stats import norm
from math import sqrt
# (b)
y = 10 # y > 10
mean_y_given_x = 10.8  
std_y_given_x = sqrt(10.24)   
prob = norm.sf(y, loc=mean_y_given_x, scale=std_y_given_x)
print(f"Probability is : {prob:.4f}")
# (c)
x = 5 # x > 5
mean_y_given_x = 4.55 
std_y_given_x = sqrt(5.76)  
prob = norm.sf(x, loc=mean_y_given_x, scale=std_y_given_x)
print(f"Probability is : {prob:.4f}")

