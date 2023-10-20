from jax.config import config
config.update("jax_enable_x64", True) 
import jax
import jax.numpy as jnp
import optax
import os
import json
from tqdm import tqdm, trange

import prk_for_optimization as IRK4
import Transformation_Functions as TFunctions
import Initial_weights
import Generate_HaltonSequence
import Convert_1D2D as convert
import Save_Results

# Initial Weights
A1, A2, B1, B2 = Initial_weights.Lobatto3A3B_4thOrder()

# Making the Halton code
halton_sequence = Generate_HaltonSequence.Halton_Sequence(150)
flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)

validation_halton = halton_sequence[100:150]
halton_sequence = halton_sequence[:100]

# # Making a 1D array
# A1D = Convert_1D2D.Convert_toOneD(A1, A2, B1, B2)
# print(A1D.shape)

# list_optimizers = [optax.adam(learning_rate)]
# opt_sgd = list_optimizers[0]
# opt_state = opt_sgd.init(A1D)
# params = A1D
# error_list_1, error_list_2 = [], []
# learning_rate, batch_size = 0.1, 100


A1D = convert.Convert_toOneD(A1, A2, B1, B2)
print(A1D.shape)

learning_rate = 0.01
list_optimizers = [optax.sgd(learning_rate)]
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
validation_error_list = []
validation_tot_error = 0
validation_avg_error = 0

flat_halton_sequence = jnp.array(halton_sequence).reshape(-1, 6)

batch_size = 100
validation_batch_size = 50

directory = 'Recorded_Results_SGD2'
if not os.path.exists(directory):
        os.makedirs(directory)
file_path1 = os.path.join(directory, 'output.txt')
# Open the file in write mode to clear it
with open(file_path1, 'w') as file:
    pass  # This does nothing, but it's enough to clear the file

def compute_grads_single(A1D, h_element):
    grad_fn = jax.jacfwd(IRK4.find_error)
    return grad_fn(A1D, h_element)

def compute_error_single(A1D, h_element):
    return IRK4.find_error(A1D, h_element)

def append_to_summary(k, avg_error):
    file_path = os.path.join(directory, f'Step_error.txt')
    
    # # Define the file name
    # file_name = '0_summary_sgd2.txt'

    # Append the data to the text file
    with open(file_path, 'a') as file:
        file.write(f'{k} : {avg_error}\n')

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
  
    """
    ------------------------------- Substitute -------------------
    """
    # tot_error = 0
    # for mm in range(len(halton_sequence)):
    #     tot_error += IRK4.find_error(A1D, halton_sequence[mm])
  
    # error_list_2.append(tot_error / len(halton_sequence))
    # A1D = A1D[:40]
    
    # # Converting A1D list, to individual A1, A2, B1 and B2
    # new_A1, new_A2, new_B1, new_B2 = Convert_1D2D.Convert_toOneD(A1D)

    # # Saving to a json file
    # Save_Results.Save_json(new_A1, new_A2, new_B1, new_B2, avg_error, k)
      
    # # Saving updated weights in different .txt files
    # Save_Results.Save_UpdatedWeights(new_A1, new_A2, new_B1, new_B2, avg_error, k)
    
    # # Saving epochs, error to a .txt file
    # Save_Results.Save_Error(avg_error, k)
    
    """
    ------------------------------- Substitute -------------------
    """

    tot_error = 0
    for mm in range(len(halton_sequence)):
        tot_error += IRK4.find_error(A1D, halton_sequence[mm])
    
    avg_error = tot_error / len(halton_sequence)
    error_list_2.append(avg_error)
    
    # Validation data set
    for mm in range(0,len(validation_halton)):
        validation_tot_error += IRK4.find_error(A1D, validation_halton[mm])
    
    validation_avg_error = validation_tot_error / len(validation_halton)
    validation_error_list.append(validation_avg_error)
    
    
    A1D = A1D[:40]
    
    ##################################################################################
    new_A1, new_A2, new_B1, new_B2 = convert.Convert_toTwoD(A1D)
    
    # Create a directory if it doesn't exist
    directory = 'Recorded_Results_SGD2'
    
    # Define the file path based on the value of k
    file_path = os.path.join(directory, f'BatchOutput_SGD2.json')
    
    # Saving to a json file.
    json_A1 = new_A1.tolist()
    json_A2 = new_A2.tolist()
    json_B1 = new_B1.tolist()
    json_B2 = new_B2.tolist()
    json_avg_error = float(avg_error)
    # Combine data into a dictionary
    data = {
        'Number' : k,
        'Error' : json_avg_error,
        'A1': json_A1,
        'A2': json_A2,
        'B1': json_B1,
        'B2': json_B2 
    }
    
    # try:
    #     with open(file_path, 'r') as file:
    #         existing_data = json.load(file)
    # except FileNotFoundError:
    #     existing_data = []

    # existing_data.append(data)
    
    # Save the data to BatchOutput.json
    with open(file_path, 'w') as file:
        json.dump(data, file)

    # print(f'Data saved to {file_path}')
    
    ###
    
    """Creating new files for all the new A1, A2, B1 and B2 values"""
    
    # Assuming you have the necessary data (new_A1, new_A2, new_B1, new_B2, avg_error, k)

    # Convert lists to strings
    A1_str = ' - '.join(map(str, new_A1.tolist()))
    A2_str = ' - '.join(map(str, new_A2.tolist()))
    B1_str = ' - '.join(map(str, new_B1.tolist()))
    B2_str = ' - '.join(map(str, new_B2.tolist()))

    # Convert avg_error to a float (if necessary)
    avg_error_float = float(avg_error) if avg_error.shape == () else avg_error.tolist() ## this line not added to the pool file.. SaveResults

    # Convert k to an integer (if it's not already)
    json_k = int(k)

    
    # Define the file path based on the value of k
    file_path = os.path.join(directory, f'output.txt')
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Read existing content (if any)
    try:
        with open(file_path, 'r') as file:
            existing_content = file.read()
    except FileNotFoundError:
        existing_content = ''


    # Write the data to the text file
    with open(file_path, 'w') as file:
        file.write(existing_content)
        file.write(f'Number: {json_k}\n')
        file.write(f'Error: {avg_error_float}\n')
        file.write(f'A1: {A1_str}\n')
        file.write(f'A2: {A2_str}\n')
        file.write(f'B1: {B1_str}\n')
        file.write(f'B2: {B2_str}\n')
        file.write('-' * 80 + '\n')  # Add a division line
    
    """Creating a single file for sequence number and error associated with it. """
    append_to_summary(k, avg_error_float)