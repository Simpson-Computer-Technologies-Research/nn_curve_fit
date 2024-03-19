##
## We'll use pytorch to fit to a curve.
##
## We'll also use matplotlib to visualize the network working in real time
## showing both the curve being fitted and the nodes in the network.
##
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##
## Create a simple neural network
##
## This network will have 3 layers and is linear
##
class Net(nn.Module):
    ##
    ## Initialize the network
    ##
    def __init__(self) -> None:
        super().__init__()

        ##
        ## Create the layers
        ##
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

        ## OR
        self.seq = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

        ##
        ## End of function: __init__
        ##

    ##
    ## Forward pass
    ##
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

        ##
        ## End of function: forward
        ##

    ##
    ## End of class: Net
    ##


##
## Train the network and visualize the results
##
if __name__ == "__main__":
    ##
    ## Create the network, optimizer, and loss function
    ##
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    ##
    ## Create the figure and axis
    ##
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ##
    ## Create a simple dataset
    ##
    x: torch.Tensor = torch.arange(-10, 10, 0.1)  ## The x-values (input)
    y: torch.Tensor = torch.sin(x)  ## The corresponding y-values (output)

    ##
    ## Define the animation function
    ##
    ## This will also train the network while visualizing the results
    ##
    def animate(_: int) -> None:
        ##
        ## Reshape the data
        ##
        x_v = x.view(-1, 1)  ## n rows, 1 column where n is automatically calculated
        y_v = y.view(-1, 1)  ## n rows, 1 column where n is automatically calculated

        ##
        ## Clear the gradients and perform a forward pass
        ##
        optimizer.zero_grad()
        output = net(x_v)

        ##
        ## Calculate the loss and backpropagate
        ##
        loss = loss_function(output, y_v)
        loss.backward()
        optimizer.step()

        ##
        ## Clear the axis and plot the data
        ##
        ax1.clear()
        ax1.plot(x, y)
        ax1.plot(x, output.detach().numpy())

    ##
    ## Run the animation
    ##
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()

##
## End of file: main.py
##
