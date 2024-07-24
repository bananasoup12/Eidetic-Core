import torch
import math
import torch.nn as nn
import random
import numpy as np
import sys
import db
import logging
np.set_printoptions(threshold=sys.maxsize)
import os

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(filename='network.log', filemode='a', level=logging.DEBUG)
logging.info("Started")

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

    def binarySearchQuantiles(self, activation, index):
        
        output_node_list_of_quantile_bounds = self.quantiles[index]
        _length = len(output_node_list_of_quantile_bounds)

        for i in range(0, _length):
                    if activation <= output_node_list_of_quantile_bounds[i]:

                        return i

        if activation > output_node_list_of_quantile_bounds[-1]:
            return len(output_node_list_of_quantile_bounds) 
        

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
    def __init__(self, size_in, size_out, num_quantiles):
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

        self.param_index = nn.ParameterList()
        self.device_to_use = os.getenv("DEVICE")

        # Intitialzing parameter list of weights to use in indexing
        for i in range(0, num_quantiles):
            for j in range(0, self.size_in):
                weights = torch.Tensor(self.size_out)
                for k in range(0, self.size_out):
                    weights[k] = self.weights[k][j]
            
                self.param_index.append(nn.Parameter(weights))

    
    def build_index(self, num_quantiles):

        #Copy weights across indices from the trained weight vector
        for i in range(0, num_quantiles):
            for j in range(0, self.size_in):
                for k in range(0, self.size_out):
                    with torch.no_grad():
                        self.param_index[i*self.size_in + j][k] = self.weights[k][j]
        
        
        
    
    def set_use_indices(self, val):
        self.use_indices = val


    #TODO: Figure out how to rewrite forward/backward pass without requiring swapping of weights
    def forward(self, x, indices):
        
        if self.use_indices == True:

            final_output = torch.empty(len(x), self.size_out).to(self.device_to_use)
            for batch in range(0, len(x)):
                indices = indices[batch]
                
                outx = torch.empty(len(x[batch]), self.size_out).to(self.device_to_use)
                
                for i,p in enumerate(x[batch]):
                    index = int(indices[i]) * self.size_in + i
                    outx[i] = self.param_index[index] * x[batch][i]
                    
            
                final_output[batch] = torch.sum(outx, 0)

            w_times_x = final_output
            
        else:
            w_times_x= torch.mm(x, self.weights.t())
            

        
        return torch.add(w_times_x, self.bias)
