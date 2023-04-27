import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set the device for computation (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters for the model and training
input_size = 28
sequence_length = 28
hidden_size = 128
output_size = 10
batch_size = 100
num_epochs = 10
learning_rate = 0.001

# Load the Fashion-MNIST dataset for training and testing
train_dataset = datasets.FashionMNIST(root='data/',
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)
test_dataset = datasets.FashionMNIST(root='data/',
                                     train=False,
                                     transform=transforms.ToTensor())

# Create DataLoader objects to load data in batches during training and testing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Define the custom RNN model
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()

        # Store the hidden layer size
        self.hidden_size = hidden_size

        # Define the input-to-hidden and input-to-output linear layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # Define the softmax activation function for the output layer
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Concatenate the input and hidden states along the feature dimension
        combined = torch.cat((input, hidden), 1)

        # Update the hidden state using the input-to-hidden linear layer
        hidden = self.i2h(combined)

        # Compute the output probabilities using the input-to-output linear layer
        output = self.i2o(combined)

        # Apply the softmax activation function to normalize the output probabilities
        output = self.softmax(output)

        # Return the output probabilities and the updated hidden state
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden state with zeros
        return torch.zeros(batch_size, self.hidden_size).to(device)

# Instantiate the custom RNN model and move it to the device
model = CustomRNN(input_size, hidden_size, output_size).to(device)

# Define the loss function (negative log likelihood loss) and the optimizer (Adam)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape the images and move them to the device
        images = images.reshape(-1, sequence_length, input_size).to(device)

        # Move the labels to the device
        labels = labels.to(device)

        # Initialize the hidden state with zeros for the current batch
        hidden = model.init_hidden(images.size(0))

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass through the sequence
        for t in range(sequence_length):
            # Update the output and hidden state at each time step
            output, hidden = model(images[:, t, :], hidden)

        # Compute the loss
        loss = criterion(output, labels)

        # Backpropagate the gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Print the progress of the training
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # Reshape the images and move them to the device
            images = images.reshape(-1, sequence_length, input_size).to(device)

            # Move the labels to the device
            labels = labels.to(device)

            # Initialize the hidden state with zeros for the current batch
            hidden = model.init_hidden(images.size(0))

            # Forward pass through the sequence
            for t in range(sequence_length):
                # Update the output and hidden state at each time step
                output, hidden = model(images[:, t, :], hidden)

            # Find the class with the highest probability in the output
            _, predicted = torch.max(output.data, 1)

            # Update the total number of samples and the number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print the test accuracy of the model
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
