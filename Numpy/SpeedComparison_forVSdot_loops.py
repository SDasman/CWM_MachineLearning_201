import numpy as np
from datetime import datetime


a = np.random.randn(100)
b = np.random.randn(100)
T = 100000


def slow_dot_product(vec1, vec2):
    result = 0
    for i, j in zip(vec1, vec2):
        result += i*j
    return result

# Slow method
t0 = datetime.now()
for x in range(0, T):
    slow_dot_product(a, b)
dt1 = datetime.now() - t0
print(dt1)

# Array method
t0 = datetime.now()
for y in range(0, T):
    a.dot(b)
dt2 = datetime.now() - t0
print(dt2)

print("dt1 / dt2 : ", dt1.total_seconds() / dt2.total_seconds())
