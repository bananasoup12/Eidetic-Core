import torch
import math
import torch.nn as nn
import random
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

class EideticLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, n_quantile_rate, quantile_cardinality):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        self.outputValues = np.zeros([quantile_cardinality + 1, size_out])
        self.index = 0
        self.n_quantile_rate = n_quantile_rate
        self.quantiles = []
        self.quantile_cardinality = quantile_cardinality
        
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def calculate_n_quantiles(self, num_quantiles):
        
        #Every val indicies in our list represent a quantile separaiton point
        val = int(self.quantile_cardinality / num_quantiles)

        #For every column is our stored list of activations
        for j in range(0, self.size_out):
            inner_quantile = []

            #Sort our list by the jth column
            self.outputValues = self.outputValues[self.outputValues[:,j].argsort(kind='mergesort')]

            #Append to our inner quantile which represents the quantiles for column j
            for i in range(0, num_quantiles-1):
                inner_quantile.append(self.outputValues[val*(i+1)][j])
            
            
            self.quantiles.append(inner_quantile)
    
    #Build index for biases
    def build_index(self, num_quantiles):
        bias = torch.Tensor(self.size_out, num_quantiles)

        #Copy my bias across my indices from the trained bias vector
        for i in range(0, len(bias)):
            for j in range(0, len(bias[i])):
                bias[i][j] = self.bias[i]
        
        self.indexed_bias = nn.Parameter(bias)


    #TODO: Convert from linear to binary search
    def binarySearchQuantiles(self, activation, index):
        
        for i in range(0, len(self.quantiles[index])):
            if activation <= self.quantiles[index][i]:
                
                return i

        if activation > self.quantiles[index][len(self.quantiles[index]) -1]:
            return len(self.quantiles[index]) 

        return -1
        
    def forward(self, x, store_activations, get_indices):
        w_times_x= torch.mm(x, self.weights.t())
        
        if store_activations == True:
            all_activations = w_times_x.detach().cpu().numpy()

            for activation_vector in all_activations:
        
                self.outputValues[self.index] = activation_vector
                self.index = self.index + 1

        indices = torch.zeros([len(w_times_x), self.size_out])
        
        if get_indices == True:
            for j in range(0, len(w_times_x)):
                for i in range(0, self.size_out):
                    indices[j][i] = self.binarySearchQuantiles(w_times_x[j][i].item(), i)


        return [torch.add(w_times_x, self.bias), indices]  

class IndexedLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
    def build_index(self, num_quantiles):
        weights = torch.Tensor(self.size_out, self.size_in, num_quantiles)

        #Copy weights across indices from the trained weight vector
        for i in range(0, len(weights)):
            for j in range(0, len(weights[i])):
                for k in range(0, len(weights[i][j])):
                    weights[i][j][k] = self.weights[i][j]

        self.indexed_weights = nn.Parameter(weights)

            
    def unfreeze_params(self):
        for param in self.indexed_weights:
            param.requires_grad = True
        
    def forward(self, x, use_indices, indices):
        w_times_x= torch.mm(x, self.weights.t())

        
        if use_indices == True:
            weights_from_index = torch.Tensor(len(x), self.size_out, self.size_in)

            for i in range(0, len(weights_from_index)):
                for j in range(0, len(weights_from_index[i])):
                    index = int(indices[i][j].item())

                    for k in range(0, len(weights_from_index[i][j])):
                        weights_from_index[i][j][k] = self.indexed_weights[j][k][index]
                
                
                w_times_x[i]= torch.matmul(x[i], weights_from_index[i].t())
            

        print("Batch")
        return torch.add(w_times_x, self.bias) 