import random

def tanh(x):
    e_pos = 2.718281828459045 ** x
    e_neg = 2.718281828459045 ** (-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

random.seed(42)

input1, input2 = 0.05, 0.10
bias1, bias2 = 0.5, 0.7

weight1 = random.uniform(-0.5, 0.5)
weight2 = random.uniform(-0.5, 0.5)
weight3 = random.uniform(-0.5, 0.5)
weight4 = random.uniform(-0.5, 0.5)
weight5 = random.uniform(-0.5, 0.5)
weight6 = random.uniform(-0.5, 0.5)
weight7 = random.uniform(-0.5, 0.5)
weight8 = random.uniform(-0.5, 0.5)

hidden1_input = input1 * weight1 + input2 * weight3 + bias1
hidden2_input = input1 * weight2 + input2 * weight4 + bias1
hidden1_output = tanh(hidden1_input)
hidden2_output = tanh(hidden2_input)

output1_input = hidden1_output * weight5 + hidden2_output * weight7 + bias2
output2_input = hidden1_output * weight6 + hidden2_output * weight8 + bias2
output1_output = tanh(output1_input)
output2_output = tanh(output2_input)

print(f"Random Weights: w1={weight1}, w2={weight2}, w3={weight3}, w4={weight4}, w5={weight5}, w6={weight6}, w7={weight7}, w8={weight8}")
print(f"Hidden Layer Outputs: h1={hidden1_output}, h2={hidden2_output}")
print(f"Output Layer Outputs: o1={output1_output}, o2={output2_output}")