import torch


class Quadrado(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # salva valores necessários para o backward
        ctx.save_for_backward(x)

        # implementa a operação de quadrado
        y = x ** 2
        return y

    @staticmethod
    def backward(ctx, dL_dy):
        (x,) = ctx.saved_tensors

        # regra da cadeia para o quadrado:
        # y = x^2  ->  dy/dx = 2x  ->  dL/dx = dL/dy * 2x
        dL_dx = dL_dy * 2 * x
        return dL_dx


# Grafo Computacional
x1 = torch.tensor(3.0, requires_grad=True)  # Entrada 1
x2 = torch.tensor(4.0, requires_grad=True)  # Entrada 2

q = torch.add(x1, x2)
q.retain_grad()
y = Quadrado.apply(q)  # y = (x1 + x2)^2

print("y =", y.item())

y.backward()
print("dL/dq =", q.grad.item())
print("dL/dx1 =", x1.grad.item())
print("dL/dx2 =", x2.grad.item())

# Verificação contra a implementação nativa do PyTorch
x1_ref = torch.tensor(3.0, requires_grad=True)
x2_ref = torch.tensor(4.0, requires_grad=True)
q_ref = torch.add(x1_ref, x2_ref)
q_ref.retain_grad()
y_ref = torch.square(q_ref)
y_ref.backward()

assert torch.isclose(y, y_ref)
assert torch.isclose(q.grad, q_ref.grad)
assert torch.isclose(x1.grad, x1_ref.grad)
assert torch.isclose(x2.grad, x2_ref.grad)
print("OK: resultados idênticos aos do torch.square")
