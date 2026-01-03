from einops import rearrange, reduce
import numpy as np


x = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24]
])

print(rearrange(x, '(h1 h2) (w1 w2) -> h2 w2 (h1 w1)', h1=2, w1=2))
