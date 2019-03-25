"""
@author: Omar M.Hussein
"""

def slashindex(filename):
    
    count = -1
    index = 0
    for i in range(len(filename)):
        count = count + 1
        if(filename[i] == '/'):
            index = count
    return index