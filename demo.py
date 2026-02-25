from micro_autograd.nn import MLP

x = [2.0, 3.0, -1.0] 
y_true = 1.0

n = MLP(3, [4, 4, 1])

print(f"totally {len(n.parameters())} in Learning \n")

print('-'*50)

for step in range(50):
    y_pred = n(x)
    loss = (y_pred - y_true)**2
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    for p in n.parameters():
        p.data -= 0.05 * p.grad
    
    print(f'time : {step+1:2d}.    output: {y_pred.data:.4f}.    loss: {loss.data:.4f}')

    
