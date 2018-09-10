import torch


c = torch.nn.CrossEntropyLoss(reduction='none')
a = torch.randn(1, 3, dtype=torch.float)
w = torch.randn(3, 2, dtype=torch.float, requires_grad=True)

y_ = torch.mm(a, w)
y = torch.empty(1, dtype=torch.long)

y[0 ] = 1
# y[1 ] = 0

softmax = torch.nn.Softmax(dim=1)

s = softmax(y_)
loss = c(y_, y)

loss_sum = torch.sum(loss)

loss_sum.backward()

grad = w.grad

print("input: {}".format(a))
print("weight: {}".format(w))
print("target: {}".format(y))
print("output: {}".format(y_))
print("softmax output: {}".format(s))
print("loss: {}".format(loss))
print("loss sum: {}".format(loss_sum))
print("w grads: {}".format(grad))
# print(loss.grads)
