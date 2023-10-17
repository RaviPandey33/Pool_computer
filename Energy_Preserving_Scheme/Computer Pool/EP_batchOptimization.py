from jax.config import config
config.update("jax_enable_x64", True) 
import jax
import jax.numpy as jnp
import optax
import json
from tqdm import tqdm, trange

import prk_for_optimization as IRK4
import Transformation_Functions as TFunctions
import Initial_weights
import Generate_HaltonSequence
import Convert_1D2D
import Save_Results

# Initial Weights
A1, A2, B1, B2 = Initial_weights.Lobatto3A3B_4thOrder()

# Making the Halton code
halton_sequence = Generate_HaltonSequence.Halton_Sequence()
flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)

# Making a 1D array
A1D = Convert_1D2D.Convert_toOneD(A1, A2, B1, B2)
print(A1D.shape)

list_optimizers = [optax.adam(learning_rate)]
opt_sgd = list_optimizers[0]
opt_state = opt_sgd.init(A1D)
params = A1D
error_list_1, error_list_2 = [], []
learning_rate, batch_size = 0.1, 100

def compute_grads_single(A1D, h_element):
    grad_fn = jax.jacfwd(IRK4.find_error)
    return grad_fn(A1D, h_element)

def compute_error_single(A1D, h_element):
    return IRK4.find_error(A1D, h_element)

# Use jax.vmap to vectorize the function over the batch
compute_grads_batched = jax.vmap(compute_grads_single, in_axes=(None, 0))
compute_error_batched = jax.vmap(compute_error_single, in_axes=(None, 0))

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
    
    # Converting A1D list, to individual A1, A2, B1 and B2
    new_A1, new_A2, new_B1, new_B2 = Convert_1D2D.Convert_toOneD(A1D)

    # Saving to a json file
    Save_Results.Save_json(new_A1, new_A2, new_B1, new_B2)
    
    # Saving epochs, error to a .txt file
    Save_Results.Save_Error(k, error)
    
    # Saving updated weights in different .txt files
    Save_Results.Save_UpdatedWeights(new_A1, new_A2, new_B1, new_B2)