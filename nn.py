#%%
#Step_1######################
'''
Initialise basic libraries
'''
############################
import math
import random


#%%
#Step_2######################
'''
Initialise a library
'''
############################
class ML_Framework:
    def __init__(self, data, _children=()):
        self.data = data
        self.value = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def plus(self, other):
        other = other if isinstance(other, ML_Framework) else ML_Framework(other)
        out = ML_Framework(self.data + other.data, (self, other))

        def _backward():
            self.value += out.value
            other.value += out.value
        out._backward = _backward
        return out

    def times(self, other):
        other = other if isinstance(other, ML_Framework) else ML_Framework(other)
        out = ML_Framework(self.data * other.data, (self, other))
        
        def _backward():
            self.value += other.data * out.value
            other.value += self.data * out.value
        out._backward = _backward
        return out

    def minus(self, other): 
        other = other if isinstance(other, ML_Framework) else ML_Framework(other)
        neg_other = ML_Framework(-other.data)
        return self.plus(neg_other)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.value = 1.0
        for node in reversed(topo):
            node._backward()

#%%
#Step_3######################
'''
Initialise our neural network
'''
############################
def squared_error_loss(model_prediction, y_desired_output):
    difference_in_prediction = model_prediction.minus(y_desired_output)
    return difference_in_prediction.times(difference_in_prediction)

class SingleLayerNeuron:
    def __init__(self, num_of_inputs):
        self.weights = [ML_Framework(0.09) for _ in range(num_of_inputs)]
        self.bias = ML_Framework(-0.9)

    def weights_bias_parameters(self):
        return self.weights + [self.bias]

    def zero_value(self):
        for p in self.weights_bias_parameters():
            p.value = 0.0
  
    def __call__(self, x):
        cumulative_sum = self.bias
        for wi, xi in zip(self.weights, x):
            product = wi.times(xi)
            cumulative_sum = cumulative_sum.plus(product)
        return cumulative_sum
#%%
#Step_4######################
'''
Initialise and train
'''
############################
bedrooms = [i for i in range(1, 4)]
prices = [100000 * i for i in bedrooms]
normalized_prices = [p/100000 for p in prices]
x_input_values = [[ML_Framework(b)] for b in bedrooms]
y_output_values = [ML_Framework(p) for p in normalized_prices]
num_of_model_inputs = len(x_input_values[0])


model = SingleLayerNeuron(1)
print("Initial weights:", [w.data for w in model.weights])
print("Initial bias:", model.bias.data)
learning_rate = 0.05
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    
    for i in range(num_of_model_inputs):

        x_model_input = x_input_values[i]
        y_desired_output = y_output_values[i]
        
        model_prediction = model(x_model_input)

        loss = squared_error_loss(model_prediction, y_desired_output)

        model.zero_value()
        loss.backward()
          
        total_loss = total_loss + loss.data
        
        for weights_bias_parameters in model.weights_bias_parameters():
            weights_bias_parameters.data = weights_bias_parameters.data - (learning_rate * weights_bias_parameters.value)

    mean_squared_error = total_loss / num_of_model_inputs

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {mean_squared_error}")
#%%
#Step_5######################
'''
Inference
'''
############################
bedroom_5 = [ML_Framework(5)]
predicted_price = model(bedroom_5)
predicted_price_denormalized = predicted_price.data * 100000
print(f"Predicted price for a 5-bedroom house: ${predicted_price_denormalized:.2f}")
# Output: Predicted price for a 5-bedroom house: $498000.00

# %%
# Initial weights: [0.09425283150136532]
# Initial bias: -0.8144629851970906