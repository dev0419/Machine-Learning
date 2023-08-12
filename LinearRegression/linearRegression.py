import numpy as np 
import matplotlib.pyplot as plt

x_train  = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(f"x_train.shape:{x_train.shape}")
m = x_train.shape
print(f"Number of training samples:{m}")

m = len(x_train)
print(f"Number of training samples: {m}")

i = 0
x_i,y_i = x_train[i],y_train[i] 
print(f"(x^({i}),y^({i})) = ({x_i},{y_i})")
w = 100 
b = 100
print(f"w:{w}\nb:{b}")

def compute_model_output(x,w,b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m) #returns a array of 0's  
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb  
        
temp_f_wb = compute_model_output(x_train,w,b)
plt.plot(x_train,temp_f_wb,c='b',label='Our Prediction')
plt.scatter(x_train,y_train,marker = 'x',c='r',label='Actual Value')
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()