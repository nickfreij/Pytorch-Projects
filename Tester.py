import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# check pytorch version NEW
torch.__version__

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Creating Data
# Since using linear model below...
weight = 0.7
bias = 0.3

# Make some X values
start = 0
end = 1
step = 0.02

# Create X and Y
X = torch.arange(start, end, step).unsqueeze(dim=1) # we need to unsqueeze
y = weight * X + bias

# Split the data into training and testing data 80% training
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Lets visualize this
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))

     # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14});



# testing this vs a linear layer model without describing the parameters explicitly
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize model parameters
        self.weights = nn.Parameter(torch.randn(1,
            requires_grad=True,
            dtype=torch.float
        ))

        self.bias = nn.Parameter(torch.randn(1,
            requires_grad=True,
            dtype=torch.float
        ))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias

# trying the linear layer model
class LinearRegressionModelv2(nn.Module):
    def __init__(self):
        super().__init__()
        # using nn.Linear() for creating the model parameters 
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         return self.linear_layer(x)
    
# Setting number of epochs (how many times passing over training data)
epochs = 1000

# Initialize loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

# Setting manual seed
torch.manual_seed(42)

# Create an instance of the model for both classes
model_0 = LinearRegressionModel()
model_1 = LinearRegressionModelv2()

# Checking parameters (multiple ways) see below
#print(list(model_0.parameters()))
#print(list(model_1.parameters()))

#for name, param in model_0.named_parameters():
#    print(f"{name}: {param.data}")

#model_1.state_dict()
#print(f'{model_0.state_dict()}')

# Let's start training and testing with a loop, but before
# We need to create our loss function, nn.L1Loss() is a good start
loss_fn = nn.L1Loss()

# Now we need an optimizer
# SGD is good for linear, we are using this to optimize the parameters of model
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# Put Data on correct device 
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Time to start the loop!
for epoch in range(epochs):
     # Train
     model_1.train()

     # 1. Forward Pass
     y_pred = model_1(X_train)

     # 2. Calculate Loss with fn
     # Difference between data y values and y values through model
     loss = loss_fn(y_pred, y_train)

     # 3. Zero grad Optimizer
     optimizer.zero_grad()

     # 4. Loss Backward
     loss.backward()

     # 5. Step the optimizer
     optimizer.step()

     # Testing
     # Puts model in eval mode for testing (can do inference mode)
     model_1.eval() 
     # 1. Forward Pass
     with torch.inference_mode():
          test_pred = model_1(X_test)

          # 2. Calculate loss
          test_loss = loss_fn(test_pred, y_test)
    
     # Print Loss every 100 epochs
#     if epoch % 100 == 0:
#        print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss} | ")

# Print Parameters of model now
#from pprint import pprint 
#print("The model learned the following values for weights and bias:")
#pprint(model_1.state_dict())
#print("\nAnd the original values for weights and bias are:")
#print(f"weights: {weight}, bias: {bias}")

# Plotting data
plot_predictions(X_train, y_train, X_test, y_test)
plot_predictions(predictions=test_pred.cpu())