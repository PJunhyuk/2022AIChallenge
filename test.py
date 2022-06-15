import numpy as np

x = np.arange(0, 256, dtype=np.float)
b=(x * 1.5) % 180
for i in range(100):
    print(np.random.choice(2,1))
print("s")
# lut_hue = ((x * r[0]) % 180).astype(np.float)