# write a program to find the inverse, rank, determinant and eigen values of a given matrix
import numpy as np

a = np.random.randint(10,size=(3,3))
print(a,'\n')
print('Inverse :\n',np.linalg.inv(a),'\n')
print('Rank :\n',np.linalg.matrix_rank(a),'\n')
print('Determinant :\n',np.linalg.det(a),'\n')
v,w = np.linalg.eig(a)
print('Eigen values :\n',v,'\n')
print('Eigen vector :\n',w,'\n')