import torch
import math
import torch.nn as nn
import random
import numpy as np
import sys
import db
np.set_printoptions(threshold=sys.maxsize)

#Testing branch protection...
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

    def calculate_n_quantiles(self, num_quantiles, use_db):
        

        if use_db == False:
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
        else:
            distribution = db.database.create_quantile_distribution(num_quantiles)

            for j in range(0, self.size_out):
                inner_quantile = []
                
                #Append to our inner quantile which represents the quantiles for column j
                for i in range(0, num_quantiles-1):
                    inner_quantile.append(distribution[j][i+1])
               
                self.quantiles.append(inner_quantile)
    
    #Build index for biases
    def build_index(self, num_quantiles):
        bias = torch.Tensor(self.size_out, num_quantiles)

        #Copy my bias across my indices from the trained bias vector
        for i in range(0, len(bias)):
            for j in range(0, len(bias[i])):
                bias[i][j] = self.bias[i]
        
        self.indexed_bias = nn.Parameter(bias)


    
    def binarySearchQuantiles(self, activation, index):
        
        low = 0
        high = len(self.quantiles[index])


        while low < high:
            mid = int(low + (high - low) / 2)

            if mid == 0:
                return 0

            if self.quantiles[index][mid] <= activation and self.quantiles[index][mid -1] >= activation:
                return mid

            if self.quantiles[index][mid] < activation:
                low = mid + 1
            else:
                high = mid - 1

        if activation > self.quantiles[index][len(self.quantiles[index]) -1]:
            return len(self.quantiles[index]) 
        

        return 0
        
    def forward(self, x, store_activations, get_indices, use_db):
        w_times_x= torch.mm(x, self.weights.t())
        
        if store_activations == True:
            all_activations = w_times_x.detach().cpu().numpy()
            
            for activation_vector in all_activations:
                
                if use_db == True:
                    
                    if self.n_quantile_rate >= random.uniform(0, 1):
                        db.database.insert_record(activation_vector)

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
        
        self.use_indices = False
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)

        bound = 1 / math.sqrt(fan_in)
        self.use_previous_indices = False
        self.previous_indices = None

        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    
    def build_index(self, num_quantiles):
        self.param_index = nn.ParameterList()
    
        #Copy weights across indices from the trained weight vector
        for i in range(0, num_quantiles):

            weights = torch.Tensor(self.size_in, self.size_out)
            
            for j in range(0, self.size_in):
                for k in range(0, self.size_out):
                    weights[j][k] = self.weights[k][j]

            self.param_index.append(nn.Parameter(weights))


            
    def unfreeze_params(self):
        for param in self.weights:
            param.requires_grad = True


    def freeze_params(self):

        for param in self.bias:
            param = param.detach()

    def unfreeze_params_by_index(self, indices):

        for i in range(0, len(self.indexed_weights)):
            for j in range(0, len(self.indexed_weights[i])):
                index = int(indices[0][j].item())
                self.indexed_weights[i][j][index].requires_grad = True
    
    def set_use_indices(self, val):
        self.use_previous_indices = True
        self.use_indices = val


    #TODO: Figure out how to rewrite forward/backward pass without requiring swapping of weights
    def forward(self, x, indices):
        
        indices = indices[0].tolist()

        #TODO: Rewrite to improve performance
        if self.use_indices == True:
            w_times_x = torch.Tensor(1, self.size_out)
            
            for i,p in enumerate(x):
                index = int(indices[i])
                w_times_x[0] = w_times_x[0] + self.param_index[index][i] * x[0][i]
                

        else:
            w_times_x= torch.mm(x, self.weights.t())
            

        
        return torch.add(w_times_x, self.bias)
