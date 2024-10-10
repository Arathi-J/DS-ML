#write a new program to display elements of the matrix x to different powers also display the identity matrix

import numpy as np

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('Identity matrix is :\n',np.identity(3, dtype= int))
print('Display each element of the matrix to different powers \n',np.power(x,[[1,2,3],[4,5,6],[7,8,9]]))