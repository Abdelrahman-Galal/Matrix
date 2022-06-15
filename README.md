# Matrix
A naive implementation of Matrix class in Python with no use of other frameworks(numpy).

The matrix class can perform basic matrices opertations like:
-  Matrix addition  (scaler and matrix addition)
-  Matrix subtraction (scaler and matrix subtraction)
-  Matrix multiplication (scaler and matrix multiplication)
-  Find transpose matrix
-  Get matrix determinant 

# Example:
```
from matrix import Matrix

A = [[6, -1], [ -1 , 3]]
B = [[7, 10], [ 1 , 2]]
A = Matrix(A)
B = Matrix(B)

# Matrix operations
print(A + B)
print(A - B)
print(A * B)

# Scaler operations
print(2 + A)
print(A + 2)
print(2 - A)
print(A - 2)
print(A * 2)
print(2 * A)

# Matrix transpose
print(B.transpose())

# Matrix determinant
A = [[6, -1, 33], [-5, 3, 66], [77, 7, 4]]
A = Matrix(A)
print(A.det())
```
