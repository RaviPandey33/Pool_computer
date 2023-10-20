import json
import os

def Save_json(new_A1, new_A2, new_B1, new_B2, avg_error, k):
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
    #     with open('BatchOutput_SGD.json', 'r') as file:
    #         existing_data = json.load(file)
    # except FileNotFoundError:
    #     existing_data = []

    # existing_data.update(data)
    
    # Save the data to BatchOutput.json
    with open('BatchOutput_SGD.json', 'w') as file:
        json.dump(data, file)

    # print(f'Data saved to {file_path}')

    
    
def Save_UpdatedWeights(new_A1, new_A2, new_B1, new_B2):
    # Saving to a json file.
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
        
    
    
def Save_Error(epoch, error):
    # Saving to a json file.
    k = epoch
    