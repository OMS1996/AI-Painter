"""
last edited on Mon Mar 24
@author: Omar M.Hussein
"""
image_file = "images/Tate.jpg"
i = len(image_file)
l1 = []
while i > 0:
  if image_file == '/':
    break
  str_append_list_join(image_file[i]) 
  i -= 1
  l1.join(l1)
print(l1)

def str_append_list_join(s, n):
    l1 = []
    i = 0
    while i < n:
        l1.append(s)
        i += 1
    return ''.join(l1)