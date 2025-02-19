
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input constants (can be adjusted as needed)
input_constant = torch.tensor([1.0, 1.0, 1.0], device=device)
data_points = torch.tensor([[2.0, 3.0, 4.0], [1.0, 0.0, -1.0]], device=device)  # Two data points in 3D
num_gaussians = 100  # Number of Gaussians
fixed_variance = torch.tensor([0.05, 0.05, 0.05], device=device)  # Fixed variance for each dimension

# Define Gaussian Mixture Model (GMM) class
class GaussianMixtureModel(nn.Module):
    def __init__(self, num_gaussians, dim):
        super(GaussianMixtureModel, self).__init__()
        self.num_gaussians = num_gaussians
        self.dim = dim
        # Parameters: means and mixing coefficients
        self.means = nn.Parameter(torch.randn(num_gaussians, dim, device=device))
        self.mixing_coeffs = nn.Parameter(torch.ones(num_gaussians, device=device) / num_gaussians)

    def forward(self, x):
        # Compute log probabilities for each Gaussian
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # Shape: (N, K, D)
        exponent = -0.5 * torch.sum((diff ** 2) / fixed_variance, dim=2)  # Shape: (N, K)
        log_gaussians = exponent - 0.5 * self.dim * torch.log(2 * torch.pi * fixed_variance).sum()

        # Compute log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(self.mixing_coeffs, dim=0)
        log_mixing_coeffs = torch.clamp(log_mixing_coeffs, min=-10)  # Prevent extreme values

        max_log = torch.max(log_gaussians + log_mixing_coeffs, dim=1, keepdim=True).values
        log_probs = max_log + torch.logsumexp(log_gaussians + log_mixing_coeffs - max_log, dim=1)
        return log_probs

# Negative log-likelihood loss
class NegativeLogLikelihoodLoss(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihoodLoss, self).__init__()

    def forward(self, log_probs):
        return -torch.sum(log_probs)

# Function to train the model
def train_model(num_epochs=2000):
    model = GaussianMixtureModel(num_gaussians, dim=3).to(device)
    criterion = NegativeLogLikelihoodLoss()
    optimizer = optim.Adam([model.means, model.mixing_coeffs], lr=0.005)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        data_point = data_points[random.randint(0, len(data_points) - 1)].unsqueeze(0)

        log_probabilities = model(data_point)
        loss = criterion(log_probabilities)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid instability
        optimizer.step()

        # Print top 3 weights every iteration
        # with torch.no_grad():
        #     mixing_weights = torch.softmax(model.mixing_coeffs, dim=0)
        #     top3_indices = torch.topk(mixing_weights, 3).indices
        #     top3_weights = mixing_weights[top3_indices]
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Top 3 Weights: {top3_weights.tolist()}")

    return model, losses

# Run training session
model, losses = train_model()

# Extract top 5 means based on mixing weights
with torch.no_grad():
    mixing_weights = torch.softmax(model.mixing_coeffs, dim=0)
    top5_indices = torch.argsort(mixing_weights, descending=True)[:5]
    top5_means = model.means[top5_indices]
    top5_weights = mixing_weights[top5_indices]

    print("\nTop 5 Means by Mixing Weights:")
    print("=" * 55)
    print(f"{'Rank':<5} {'Mean Values':<40} {'Weight':<10}")
    print("=" * 55)

    for i in range(5):
        mean_values = ", ".join(f"{v:.4f}" for v in top5_means[i].cpu().numpy())
        print(f"{i+1:<5} [{mean_values:<40}] {top5_weights[i].item():.4f}")

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Joint Optimization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
 