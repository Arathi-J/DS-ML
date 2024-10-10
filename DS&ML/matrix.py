import numpy as np

a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
a2 = np.array([[7,8,9],[4,5,6],[1,2,3]])
print('First matrix :')
print(a1,'\n')
print('Second matrix :')
print(a2,'\n')
print('Matrix Addition :')
print(np.add(a1,a2),'\n')
print('Matrix Subtraction :')
print(np.subtract(a1,a2),'\n')
print('Matrix Multiplication :')
print(np.multiply(a1,a2),'\n')
print('Matrix Division :')
print(np.divide(a1,a2),'\n')
print('Transpose of first matrix :')
print(a1.T,'\n')
print('Transpose of second matrix :')
print(a2.T,'\n')
print('Sum of diagonal elements :')
print(np.add(np.diag(a1),np.diag(a2)),'\n')



