import numpy as np

def test_matrix_multiplication():
    print("Test 1: 2x2 multiplied by 2x2")
    t1 = np.array([1, 2, 3, 4]).reshape(2, 2)
    t2 = np.array([5, 6, 7, 8]).reshape(2, 2)
    print(t1 @ t2)

    print("\nTest 2: 2x3 multiplied by 3x2")
    t3 = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
    t4 = np.array([7, 8, 9, 10, 11, 12]).reshape(3, 2)
    print(t3 @ t4)

    print("\nTest 3: Incompatible shapes (should raise error)")
    try:
        t5 = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        t6 = np.array([7, 8, 9]).reshape(3, 1)
        print(t5 @ t6)
    except ValueError as e:
        print("Error:", e)

    print("\nTest 4: Higher dimensional tensors")
    t7 = np.random.rand(2, 3, 4)
    t8 = np.random.rand(2, 4, 3)
    print(t7 @ t8)

    print("\nTest 5: Vector and Matrix")
    t9 = np.array([1, 2, 3])
    t10 = np.array([[4, 5], [6, 7], [8, 9]])
    print(t9 @ t10)

test_matrix_multiplication()