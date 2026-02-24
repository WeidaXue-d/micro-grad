from micro_autograd.engine import Value

a= Value (2.0)
b= Value (3.0)
c= Value (4.0)


d = a * b
y = d + c

y.backward()

print("y_grad:", y.grad)
print("c_grad:", c.grad)
print("d_grad:", d.grad)
print("a_grad:", a.grad)
print("b_grad:", b.grad)


'''
print('-'*30)
print('Before')

print('a_grad:',a.grad)
print('b_grad:',b.grad)

y.grad=1.0
y._backward()
d._backward()

print('-'*30)
print("After")
print("y_grad:", y.grad)
print("c_grad (from plus):", c.grad)
print("d_grad (from plus):", d.grad)
print("a_grad (from multi d=a*b, b=3.0):", a.grad)
print("b-grad (from multi d=a*b, a=2.0):", b.grad)

d= a * b
y= d + c

print('y_val:',y)
print('y_parents',y._prev)
print('y_op',repr(y._op)) 

print('-'*30)

print('d_val',d)
print('d_parent',d._prev)
print('d_op',repr(d._op))

print('a:', a)
print('b:', b)

c= a + b
print('c:',c)
print('c_parents:',c._prev)
print('c_op:',repr(c._op))

#functions of values
print('a_data:', a.data)
print('b_data:', b.data)    

print('a_grad:', a.grad)
print('b_grad:', b.grad)

print('-'*30)

#source of values
print('a_prev:', a._prev)
print('b_prev:', b._prev)
print('a_op:', repr(a._op))
print('b_op:', repr(b._op))
'''


