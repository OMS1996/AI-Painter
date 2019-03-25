
"""
Created specifcially to obtain the images from your Hard-Drive
Created on Sun
@author: OmarMohamedHussein
"""

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image
import os

def slashindex(filename):
    
    count = -1
    index = 0
    for i in range(len(filename)):
        count = count + 1
        if(filename[i] == '/'):
            index = count
    return index

root = tk.Tk().withdraw() 
file_path = filedialog.askopenfilename()
content_path = file_path
start_index = slashindex(content_path)
final_index = len(content_path)
pname = str(content_path[start_index:final_index])

#Path of the image
image_file = pname
#Open the image
img_org = Image.open(image_file)
# get the size of the original image
width_org, height_org = img_org.size
#The Dimensions of the picture
width = 300
height = 225

#validation
if( (width_org == width) & (height_org == height) ):
    print("IT's Already",width,"*",225 ,"no need to change")
#Transformation
else:
    # best down-sizing filter
    img_anti = img_org.resize((width, height), Image.ANTIALIAS)

    # split image filename into name and extension
    name, ext = os.path.splitext(image_file)

    # create a new file name for saving the result
    new_image_file = "%s%s%s" % (name, "_resized_300*225om", ext)
    img_anti.save(new_image_file)
    print("resized file saved as %s" % new_image_file)
    print("ALL DONE !")