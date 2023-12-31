import numpy as np

a = np.array([1, 2, 3, 4]).reshape(2, 2)
b = np.array([5, 6]).reshape(2)
c = np.array([7, 8]).reshape(1, 2)

# Test 1: a + b
print(a + b)

# Test 2: a - c
print(a - c)

# Test 3: b * a
print(b * a)

# Test 4: c / a
print(c / a)