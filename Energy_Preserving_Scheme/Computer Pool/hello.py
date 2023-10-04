import os
print(os.getcwd())

import json


# Open the existing JSON file in read mode
with open('hello_output.json', 'r') as file:
    numbers = json.load(file)

# Append new numbers to the list
for i in range(100, 1000):
    numbers.append(i)

# Write the updated list back to the JSON file
with open('hello_output.json', 'w') as file:
    json.dump(numbers, file)
