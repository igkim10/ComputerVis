#the goal is to create a library by inputting an image filename, and then prompting the user to enter the key type or name of the image
from PIL import Image
import json
import os

# create empty dictionary
Buildings = {}
def fill_Library():
# input the name of folder containing images
    folderName = input("enter filename containing images ")
    directory = '/Users/iankim/Desktop/Python_proj/p6-computervis/' + folderName
# open for user
    for fileName in os.listdir(directory):
        im = Image.open(directory +'/'+ fileName)
        im.show()
# prompt user if they know name of building 
        buildingName = input("What is the name of this building? ")
# add the entry to the dictionary
        Buildings[fileName] = buildingName
# prompt user if they have another image
    answer = input('do you have another folder of images?')
# if answer is yes, add another entry to dictionary
    if(answer == 'yes'):
        fill_Library()
# if no, save the current dictionary
    else:
        print('thank you')
        print(Buildings)
        CityName = input('what city are these images from?')
        fileName1 = CityName + ('.txt')
        with open(fileName1,'a+') as file:
            file.write(json.dumps(Buildings))


def main():
    print(Buildings)
    fill_Library()



if __name__ == "__main__":
    main()


