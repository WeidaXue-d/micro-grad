from micro_autograd.nn import Neuron

x = [2.0, 3.0]
y_true = 1.0

n = Neuron(2)


print(f"1st output{n(x).data:.4f}")
print('-'*30,'traning','-'*30)

for step in range(60):
    y_pred = n(x)

    loss = (y_pred + -1.0) * (y_pred + -1.0)

    for p in n.w + [n.b]:
        p.grad = 0.0

    loss.backward()

    for p in n.w + [n.b]:
        p.data -= 0.1 * p.grad

    print(f"time: {step+1:2d} | result: {y_pred.data:.4f} | Loss: {loss.data:.4f}")

print('-'*30,'finish','-'*30)

print("input [2.0, 3.0],output:", n(x).data)

print("w:", [p.data for p in n.w])
print("b:", n.b.data)