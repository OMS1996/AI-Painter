"""
last edited on Sun Mar 24 2019
@author: Omar M.Hussein
"""

from PIL import Image
import os

#Path of the image
image_file = "images/Tate.jpg"
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
    new_image_file = "%s%s%s" % (name, "_resized_300*225", ext)
    img_anti.save(new_image_file)
    print("resized file saved as %s" % new_image_file)
    print("ALL DONE !")