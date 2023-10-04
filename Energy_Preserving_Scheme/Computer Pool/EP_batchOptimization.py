import jax
from jax.config import config
config.update("jax_enable_x64", True) 

import jax.numpy as jnp
import optax
import json
import sys
from skopt.space import Space
from skopt.sampler import Halton
from jax import jacfwd

# Special Transform Functions
from jax import grad, jit, vmap, pmap
import jax
from jax import jit

from jax._src.lax.utils import (
    _argnum_weak_type,
    _input_dtype,
    standard_primitive,)
from jax._src.lax import lax

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import prk_for_optimization as IRK4
import Transformation_Functions as TFunctions


# Lobatto 3A and B fourth order
A1 = jnp.array([
     [0., 0., 0., 0.],
     [5/24, 1/3, -1/24, 0.],
     [1/6, 2/3, 1/6, 0.],
     [0., 0., 0., 0.]])
B1 = jnp.array([1/6, 2/3, 1/6, 0.])

# Lobatto IIIB fourth-order
A2 = jnp.array([
     [1/6, -1/6, 0., 0.],
     [1/6, 1/3, 0., 0.],
     [1/6, 5/6, 0., 0.],
     [0., 0., 0., 0.]])
B2 = jnp.array([1/6, 2/3, 1/6, 0.])

## Lobatto IIIB methods are A-stable, but not L-stable and B-stable.

## Making the Halton code

spacedim = [(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5),(-1.0, 0.5) ]

space = Space(spacedim)

halton = Halton()
n = 100

halton_sequence = halton.generate(space, n)
halton_sequence = jnp.array(halton_sequence)


A1D = TFunctions.One_Dim_Matrix(A1)
A1D = TFunctions.Add_B_tomatrix_A(A1D, B1)
A2D = TFunctions.One_Dim_Matrix(A2)
A2D = TFunctions.Add_B_tomatrix_A(A2D, B2)
A1D = TFunctions.Add_B_tomatrix_A(A1D, A2D)
print(A1D.shape)

learning_rate = 0.1
list_optimizers = [optax.adam(learning_rate)]
# chosing Stochastic Gradient Descent Algorithm.
# # We have created a list here keeping in mind that we may apply all the optimizers in optax by storing their objects in the list
 
opt_sgd = list_optimizers[0]
opt_state = opt_sgd.init(A1D)

params = A1D

count = 0
data_epoc = 10
data_epoc_list = []
repetetion = 10
# length of halton sequence = 10 

tot_eror = 0
error_list_1 = [] 
error_list_2 = []

flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)

batch_size = 100

def compute_grads_single(A1D, h_element):
    grad_fn = jax.jacfwd(IRK4.find_error)
    return grad_fn(A1D, h_element)

def compute_error_single(A1D, h_element):
    return IRK4.find_error(A1D, h_element)

# Use jax.vmap to vectorize the function over the batch
compute_grads_batched = jax.vmap(compute_grads_single, in_axes=(None, 0))
compute_error_batched = jax.vmap(compute_error_single, in_axes=(None, 0))

error_list_1 = [] 
error_list_2 = []
tot_error = 0
for k in trange(10):
    tot_error = 0

    for batch_idx in range(0, len(flat_halton_sequence), batch_size):
        batch_halton = flat_halton_sequence[batch_idx:batch_idx + batch_size]
        gradF = compute_grads_batched(A1D, batch_halton)

        avg_gradF = jnp.mean(gradF, axis=0)
        updates, opt_state = opt_sgd.update(avg_gradF, opt_state)

        A1D = optax.apply_updates(A1D, updates)
        batch_error = jnp.mean(compute_error_batched(A1D, batch_halton))
        tot_error += batch_error

    avg_error = tot_error / (len(flat_halton_sequence) // batch_size)
    error_list_1.append(avg_error)
  
    tot_error = 0
    for mm in range(len(halton_sequence)):
        tot_error += IRK4.find_error(A1D, halton_sequence[mm])
  
    error_list_2.append(tot_error / len(halton_sequence))

    A1D = A1D[:40]
    
    new_A1, new_B1 = TFunctions.actual_A_1D(A1D[0:20])
    new_A2, new_B2 = TFunctions.actual_A_1D(A1D[20:40])

    #converting A to a 2D Array
    new_A1 = TFunctions.One_D_to_TwoD(new_A1)
    new_A2 = TFunctions.One_D_to_TwoD(new_A2)
    
    
    # Saving to a json file.
    json_A1 = new_A1.tolist()
    json_A2 = new_A2.tolist()
    json_B1 = new_B1.tolist()
    json_B2 = new_B2.tolist()

    # Combine data into a dictionary
    data = {
        'A1': json_A1,
        'A2': json_A2,
        'B1': json_B1,
        'B2': json_B2
    }
    
    file_path = 'BatchOutput.json'

    # Save the data to BatchOutput.json
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)