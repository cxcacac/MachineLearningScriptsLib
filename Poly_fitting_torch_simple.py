# -*- coding: utf-8 -*-
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3)
# so we can consider it as a linear layer neural network.
# broadcasting semantics will apply to obtain a tensor of shape (2000, 3) 
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# Use the nn package to define our model as a sequence of layers. 
# nn.Sequential is a Module which contains other Modules 
# The Linear Module computes output from input using a linear function
# The Flatten layer flatens the output of the linear layer to match the shape of `y`.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

learning_rate = 1e-6
for t in range(2000):

    # Forward pass: compute predicted y by passing x to the model. 
    # Module objects override the __call__ operator so you can call them like functions. 
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # compute gradient of the loss with respect to all the learnable parameters of the model. 
    loss.backward()

    # Update the weights using gradient descent.
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad
    optimizer.step()

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')