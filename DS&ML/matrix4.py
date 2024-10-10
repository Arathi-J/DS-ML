# Write a Program to display various elements of a given 4x4 matrix  specifying appropriate indices
import numpy as np

a=np.array([[7,3,2,1],[5,6,7,8],[9,3,2,1],[4,1,2,6]])
print(a,'\n')
print("excluding first row :\n", a[1:,:],'\n')


print("excluding last column :\n", a[:,:-1],'\n')


print("elements of first and second columns in 2nd and 3rd row :\n", a[1:3,:2],'\n')


print("elements of 2nd and third columns :\n", a[:,1:3],'\n')


print("second and third element of first row :\n", a[:1,1:3],'\n')

ele = a.reshape(-1)[4:10]
print("elements from indices 4 to 10 in descending order :\n",-np.sort(-ele),'\n')
