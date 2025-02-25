import random as rnd

def tanh(x):
    return (2.718281828459045 ** x - 2.718281828459045 ** (-x)) / (2.718281828459045 ** x + 2.718281828459045 ** (-x))

rnd.seed(42)

i1, i2 = 0.05, 0.10
b1, b2 = 0.5, 0.7

w = [rnd.uniform(-0.5, 0.5) for _ in range(8)]

h1_in = i1 * w[0] + i2 * w[2] + b1
h2_in = i1 * w[1] + i2 * w[3] + b1
h1_out, h2_out = tanh(h1_in), tanh(h2_in)

o1_in = h1_out * w[4] + h2_out * w[6] + b2
o2_in = h1_out * w[5] + h2_out * w[7] + b2
o1_out, o2_out = tanh(o1_in), tanh(o2_in)

print(f"Random Weights: w1={w[0]}, w2={w[1]}, w3={w[2]}, w4={w[3]}, w5={w[4]}, w6={w[5]}, w7={w[6]}, w8={w[7]}")
print(f"Hidden Layer Outputs: h1={h1_out}, h2={h2_out}")
print(f"Output Layer Outputs: o1={o1_out}, o2={o2_out}")
