import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class VariationalAutoencoder(nn.Module):
    def __init__(self, image_size, hidden_dimension, latent_dimension):
        super(VariationalAutoencoder, self).__init__()

        self.image_size = image_size

        # Using linear (fully connected) layers but may change to convolutional layers
        # Bnn.Linear(input_layer_size, output_layer_size)
        # Encoder - Create Encoder Layers
        self.fc1 = nn.Linear(image_size, hidden_dimension)
        self.fc2_mu = nn.Linear(hidden_dimension, latent_dimension)
        self.fc2_var = nn.Linear(hidden_dimension, latent_dimension)  # Gonna use log variance

        # Decoder - Create Decoder Layers
        self.fc3 = nn.Linear(latent_dimension, hidden_dimension)
        self.fc4 = nn.Linear(hidden_dimension, image_size)


    def encode(self, x:torch.Tensor):
        print("encoding")
        h1_relu = torch.relu(self.fc1(x))
        return self.fc2_mu(h1_relu), self.fc2_var(h1_relu)

    def reparameterize(self, mu, var):
        print("reparamaterizing")
        std = torch.exp(0.5 * var)  # We use this instead of torch.std for differentiability
        eps = torch.randn_like(std)  # noise
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        print("decoding")
        h3_relu = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3_relu))  # we use sigmoid to make it a probability

    def forward(self, x: torch.Tensor):
        flatten_image = x.view(-1, self.image_size)
        mu, var = self.encode(flatten_image)
        z = self.reparameterize(mu, var)  # Latent Space Sample
        return self.decode(z), mu, var


def loss_error(reconstructed_x, x, mu, var, image_size):
    """
    We use the KL Divergence Loss to test how close our output is to the distribution
    and use Binary Cross entropy for reconstruction loss
    """
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    BCE = nn.functional.binary_cross_entropy(reconstructed_x, x.view(-1, image_size), reduction='sum')
    return BCE + KLD


def flatten_image(x):
    return x.view(-1)

def run():
    # Hyper Parameters - All these number change later
    latent_dimension = 10
    hidden_dimension = 30  # Technically should be way higher for the amount of pixels i have but sooo slow
    image_size = 640 * 640
    batch_size = 64  # For Batch gradient Descent, change num later,
    learning_rate = 1e-3
    num_epochs = 3

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Convert images to tensors and flatten them
    transform = transforms.Compose([
        transforms.Resize((640, 640 )),
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(flatten_image)  # Flatten the images
    ])
    print("images converted to tensors!")

    # Load car dataset
    # UPDATE ROOT
    train_dataset = datasets.MNIST(root='./Vehicle_Detection_Image_Dataset/train/less_images', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("dataset loaded!")

    model = VariationalAutoencoder(image_size, hidden_dimension, latent_dimension).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Training Loop
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        print("epoch: " + str(epoch))
        train_loss = 0  # Initialize training loss for the epoch
        for batch_idx, (data, _) in enumerate(train_loader):
            print(len(train_loader))
            print("batch index: " + str(batch_idx))
            data = data.to(device)
            optimizer.zero_grad()  # Zero the gradients
            recon_batch, mu, var = model(data)  # Forward pass
            loss = loss_error(recon_batch, data, mu, var, image_size)  # Calculate loss
            loss.backward()  # Backward pass
            train_loss += loss.item()  # Accumulate training loss
            optimizer.step()  # Update model parameters

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')  # Print average loss for the epoch

    # Generate new images using the trained VAE
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        sample = torch.randn(64, latent_dimension).to(device)  # Sample from standard normal distribution
        generated_images = model.decode(sample).cpu()  # Generate images from the latent space

    # Reshape and save generated images
    generated_images = generated_images.view(64, 1, 640, 640)  # Reshape to image dimensions
    torchvision.utils.save_image(generated_images, 'generated_images.png')  # Save generated images

run()
