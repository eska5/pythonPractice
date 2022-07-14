from sqlalchemy import false
import torch
import torch.nn as nn
import numpy as np

# tensorBasics
def tensorBasics():
    x = torch.rand(4, 4)
    y = torch.ones(4, 4)
    y.add_(x)
    print(y)
    print(x[1, 2].item())
    xNum = x.numpy()
    print(xNum)

    a = np.ones(4)
    print(a)
    b = torch.from_numpy(a)
    print(b)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("device is available" + str(device))


# gradients
def gradientsBasics():
    x = torch.randn(3, requires_grad=True)
    print(x)

    y = x + 2
    print(y)
    z = y * y * 2
    # z = z.mean()
    print(z)
    v = torch.tensor([0.1, 0.3, 0.8], dtype=torch.float32)
    z.backward(v)  # dz/dx
    # print(x.grad)


# autograd
def autogradBasics():
    x = torch.randn(3, requires_grad=True)
    print(x)
    # x.requires_grad_(false)
    # u can also use x.detach()
    # or
    # with torch.no_grad():
    print(x)
    weights = torch.ones(4, requires_grad=True)

    # optimizer = torch.optim.SGD(weights, lr=0.01)
    # optimizer.step()
    # optimizer.zero_grad()

    # for epoch in range(3):
    #     model_output = (weights*3).sum()
    #     model_output.backward()
    #     print(weights.grad)
    #     weights.grad.zero_()


# backPropagation
def backPropagation():
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)

    w = torch.tensor(1.0, requires_grad=True)

    # forward pass and compute the loss
    y_hat = w * x
    loss = (y_hat - y) ** 2

    print(loss)

    # backward pass
    loss.backward()
    print(w.grad)

    # UPDATE weights
    # next forward and backward pass


# gradient descent using Numpy
def gradientsNumpy():
    X = np.array([1, 2, 3, 4], dtype=np.float32)
    Y = np.array([2, 4, 6, 8], dtype=np.float32)

    w = 0.0

    # model prediction
    def forward(x):
        return w * x

    # loss
    def loss(y, y_predicted):
        return ((y_predicted - y) ** 2).mean()

    # gradient
    # MSE = 1/N * (2*x - y)**2
    # dJ/dw = 1/N 2x (w*x - y)
    def gradient(x, y, y_predicted):
        return np.dot(2 * x, y_predicted - y).mean()

    print(f"Prediction before training: f(5) = {forward(5):.3f}")

    # Training
    learning_rate = 0.01
    n_iters = 20

    for epoch in range(n_iters):
        # prediction = forward pass
        y_pred = forward(X)

        # loss
        l = loss(Y, y_pred)

        # gradients
        dw = gradient(X, Y, y_pred)

        # update weights
        w -= learning_rate * dw

        if epoch % 2 == 0:
            print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

    print(f"Prediction before training: f(5) = {forward(5):.3f}")


# =======================================================

# gradient descent using Torch
def gradientsTorch():

    # 1) Design model (input, output size, forward pass)
    # 2) Construction loss and optimizer
    # 3) Training loop
    #    - forward pass: compute prediction
    #    - backward pass: gradients
    #    - update weights

    X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

    X_text = torch.tensor([5], dtype=torch.float32)
    n_samples, n_features = X.shape
    print(n_samples, n_features)

    input_size = n_features
    output_size = n_features

    # model = nn.Linear(input_size, output_size)

    class LinearRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__init__()
            # define layers
            self.lin = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.lin(x)

    model = LinearRegression(input_size, output_size)

    print(f"Prediction before training: f(5) = {model(X_text).item():.3f}")

    # Training
    learning_rate = 0.01
    n_iters = 100

    loss = nn.MSELoss()  # mean square error (y'-y)^2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_iters):
        # prediction = forward pass
        y_pred = model(X)

        # loss
        l = loss(Y, y_pred)

        # gradients = backward pass
        l.backward()  # dl/dw

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, b] = model.parameters()
            print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

    print(f"Prediction before training: f(5) = {model(X_text).item():.3f}")


# --Main--
# tensorBasics()
# gradientsBasics()
# autogradBasics()
# backPropagation()
# gradientsNumpy()
gradientsTorch()
