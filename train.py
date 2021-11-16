import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
from torch.utils.data import Dataset
from torch import optim




# Class to access the processed training data
# Is a sublcass of torch's Dataset
class ChessValueDataset(Dataset):

  # Constructor; called when an object is initialized
  def __init__(self):
    # The training data is loaded
    dat = np.load("processed/dataset.npz")
    # The features
    self.X = dat['arr_0']
    # The labels
    self.Y = dat['arr_1']
    print("loaded", self.X.shape, self.Y.shape)

  # Function to return the length of the training data
  def __len__(self):
    return self.X.shape[0]

  # Function to return the nth entry in the training data
  def __getitem__(self, idx):
    return (self.X[idx], self.Y[idx])

# Class to initialize the neural network
# Is a sublcass of torch's Module
class Net(nn.Module):

  # Constructor; called when an object is initialized
  def __init__(self):

    # Calls the constructor of the parent class; Module
    super(Net, self).__init__()

    # Initialize the convolutional layers
    self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
    self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

    self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

    self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
    self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
    self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

    self.d1 = nn.Conv2d(128, 128, kernel_size=1)
    self.d2 = nn.Conv2d(128, 128, kernel_size=1)
    self.d3 = nn.Conv2d(128, 128, kernel_size=1)

    # Initialize the fully connected layer
    self.last = nn.Linear(128, 1)

  # Function to assign activation function for each layer
  # x; input data
  def forward(self, x):

    x = F.relu(self.a1(x))
    x = F.relu(self.a2(x))
    x = F.relu(self.a3(x))

    x = F.relu(self.b1(x))
    x = F.relu(self.b2(x))
    x = F.relu(self.b3(x))

    x = F.relu(self.c1(x))
    x = F.relu(self.c2(x))
    x = F.relu(self.c3(x))

    x = F.relu(self.d1(x))
    x = F.relu(self.d2(x))
    x = F.relu(self.d3(x))

    # Reshape the data
    x = x.view(-1, 128)
    x = self.last(x)

    # Assigns tanh activation function to the last layer
    return tanh(x)



# Main function
if __name__ == "__main__":

  # Lets the model be trained on the GPU
  device = "cuda"

  # Assigns the number of epochs for training
  NUM_EPOCHS = 100

  # Initialize an object to load training data
  chess_dataset = ChessValueDataset()

  # Convert training data into torch's DataLoader class
  train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)

  # Initialize the neural network
  model = Net()

  # Set Adam as the optimizer
  optimizer = optim.Adam(model.parameters())

  # Set MSE as the loss function
  floss = nn.MSELoss()

  # Assign CUDA i.e GPU support, to the model
  if device == "cuda":
    model.cuda()

  # Start training the model
  model.train()

  # Iterating over each epoch
  for epoch in range(NUM_EPOCHS):

    # Initialize initial loss to 0
    all_loss = 0
    num_loss = 0

    # Iterating over each entry in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):

      # Create the tensor  
      target = target.unsqueeze(-1)

      # Move the data to GPU
      data, target = data.to(device), target.to(device)

      # Convert the data to floating point
      data = data.float()
      target = target.float()

      # Set gradient to 0 before back-propogation
      optimizer.zero_grad()

      # Obtain output from model
      output = model(data)

      # Calculate the loss
      loss = floss(output, target)

      
      # Back-propogation
      loss.backward() # Calculate loss gradient
      optimizer.step() # Update the parameters
      
      # Compute the average loss
      all_loss += loss.item()
      num_loss += 1
      avg_loss = all_loss/num_loss

    print("%3d: loss=%f" % (epoch, avg_loss))

    # Save the model in nets/ directory    
    torch.save(model.state_dict(), "nets/value.pth")

