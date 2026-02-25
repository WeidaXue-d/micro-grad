from micro_autograd.nn import MLP

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] 

n = MLP(3, [4, 4, 1])
print(f"tottally {len(n.parameters())} in learning。\n")
print("================================")

for step in range(100):
    

    ypred = [n(x) for x in xs]
    
    
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    
    for p in n.parameters():
        p.data -= 0.05 * p.grad
        
    
    if step % 10 == 0 or step == 99:
        print(f" {step:3d} time | (Loss): {loss.data:.4f}")

print("================================\n")


print("final：")
for i, x in enumerate(xs):
    print(f"time :{i+1} : {x} | standard: {ys[i]:5.1f} | output: {n(x).data:.4f}")