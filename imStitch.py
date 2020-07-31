import imageio
import numpy as np
import os
from matplotlib import pyplot as plt

def combine_prokudin(imName):
     file = '/Users/iankim/Desktop/Python_proj/p6-computervis/prokudin-gorskii/' + imName
     img = imageio.imread(file)
     num_rows = np.shape(img)[0]
     num_cols = np.shape(img)[1]
     new_rows = int(num_rows/3)

     blue = img[0:new_rows,0:num_cols]
     green = img[new_rows:2*new_rows,0:num_cols]
     red = img[new_rows*2:num_rows,0:num_cols]

     plt.imshow(red)
     plt.imshow(green)
     plt.imshow(blue)

     plt.show()


def standardize(array): 
     mean = np.mean(array)
     std = np.std(array)
     return (array - mean)/std


def image_patches(image,patch_size=(16, 16)):
     file = '/Users/iankim/Desktop/Python_proj/p6-computervis/' + image
     img = imageio.imread(file)
     
     num_patches_horizontal = int(np.shape(img)[1]/patch_size[1])
     num_patches_vertical = int(np.shape(img)[0]/ patch_size[0])
     stitches = []
     i = 0
     j = 0
     for i in range(num_patches_vertical):
          for j in range(num_patches_horizontal):
             stitch = img[j:j+patch_size[0], i:i+patch_size[1]]
             stitch = standardize(stitch) 
             stitches.append(stitch)
             j += patch_size[1]
          i += patch_size[0]
     print(stitches)
     

#DOES WORK
def avg_filter(image,x_pos,y_pos):
     array = image[(x_pos -1) : (x_pos + 1), (y_pos-1): (y_pos + 1)]
     avg = int(np.average(array))
     return avg 

def filter_image(image):
     file = '/Users/iankim/Desktop/Python_proj/p6-computervis/' + image
     img = imageio.imread(file)
     num_cols = np.shape(img)[1]
     num_rows = np.shape(img)[0]
     for i in range(1,num_rows - 1):
          for j in range(1,num_cols - 1):
               img[i][j] = avg_filter(img,i,j) 
     return img

     
def sharpen_image(image):
     file = '/Users/iankim/Desktop/Python_proj/p6-computervis/' + image
     img = imageio.imread(file) 
     details = filter_image(image) - img
     img =  img - details
     plt.imshow(img,cmap=plt.get_cmap("gray"))
     plt.show()

def find_edges(image) :
     img = filter_image(image)
     gradient_x  =  np.gradient(img)[0]
     gradient_y  = np.gradient(img)[1]
     plt.imshow(gradient_x,cmap = plt.get_cmap("gray"))
     plt.imshow(gradient_y,cmap = plt.get_cmap("gray"))
     plt.show()
    



   
def main():
     name = input('enter name of img')     
     #combine_prokudin(name)
     #image_patches(name)
     #sharpen_image(name)
     find_edges(name)
     

if __name__ == "__main__":
    main()
