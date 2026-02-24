from micro_autograd.engine import Value

from micro_autograd.engine import Value


x1 = Value(2.0)
x2 = Value(0.0)


w1 = Value(-3.0)
w2 = Value(1.0)

# 偏置 b 
b = Value(6.8819727678234)

# (x1 * w1) + (x2 * w2) + b
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b

o = n.tanh()
o.backward()

print("-"*30)
print("final_o:", o.data)

print("-"*30)
print("to increase the result，w1_weight:", w1.grad)
print("to increase the result，w2_weight:", w2.grad)