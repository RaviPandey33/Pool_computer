import json

def Save_json(new_A1, new_A2, new_B1, new_B2):
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
    
    try:
        with open('BatchOutput_SGD.json', 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []

    existing_data.update(data)
    
    # Save the data to BatchOutput.json
    with open('BatchOutput_SGD.json', 'w') as file:
        json.dump(existing_data, file)

    # print(f'Data saved to {file_path}')
    
def Save_Error(epoch, error):
    # Saving to a json file.
    
def Save_UpdatedWeights(new_A1, new_A2, new_B1, new_B2):
    # Saving to a json file.