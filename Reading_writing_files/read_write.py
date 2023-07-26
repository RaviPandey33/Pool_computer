

# with open ("data.txt", "r") as myfile:
#     data = myfile.read().splitlines()


# text = "Ravi" #["Ravi", "rakesh", "rudransh", "rahul", "janki", "dinesh", "janma"]

# file1 = open("MyFile1.txt","w")

# file1.write(text)

# file1.close()


### Geeks for geeks

file1 = open("data.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n", "This is Konark \n", "This is Ranthambhore \n"]
 
# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close() #to change file access modes
 
# file1 = open("data.txt","r+")
 
# print("Output of Read function is ")
# print(file1.read())
# print()