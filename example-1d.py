import numpy as np
import matplotlib.pyplot as plt
import torch

import lightning
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datasets import SineData
from math import pi

# Create dataset
dataset = SineData(amplitude_range=(-1., 1.),
                   shift_range=(-.5, .5),
                   num_samples=2000)

# Visualize data samples
for i in range(64):
    x, y = dataset[i] 
    plt.plot(x.numpy(), y.numpy(), c='b', alpha=0.5)
    plt.xlim(-pi, pi)

plt.show()

from neural_processes import NeuralProcess

# from neural_process import NeuralProcess

x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)

# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.Tensor(np.linspace(-pi, pi, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)

for i in range(64):
    z_sample = torch.randn((1, z_dim))  # Shape (batch_size, z_dim)
    # Map x_target and z to p_y_target (which is parameterized by a 
    # normal with mean mu and std dev sigma)
    dist = neuralprocess.target_decoder(x_target, z_sample)
    # Plot predicted mean at each target point (note we could also
    # sample from distribution but plot mean for simplicity)
    plt.plot(x_target.numpy()[0], dist.loc.detach().numpy()[0], 
             c='b', alpha=0.5)
    plt.xlim(-pi, pi)

plt.show()

from torch.utils.data import DataLoader
# from training import NeuralProcessTrainer


batch_size = 20
num_context = 4
num_target = 4

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
# np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
#                                   num_context_range=(num_context, num_context),
#                                   num_extra_target_range=(num_target, num_target), 
#                                   print_freq=200)

# neuralprocess.training = True
# np_trainer.train(data_loader, 30)

trainer = lightning.Trainer(accelerator="cuda", max_epochs=10)
trainer.fit(model=neuralprocess, train_dataloaders=data_loader)


x_target = torch.Tensor(np.linspace(-pi, pi, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)

for i in range(64):
    z_sample = torch.randn((1, z_dim))  # Shape (batch_size, z_dim)
    # Map x_target and z to p_y_target (which is parameterized by a 
    # normal with mean mu and std dev sigma)
    dist = neuralprocess.target_decoder(x_target, z_sample)
    # Plot predicted mean at each target point (note we could also
    # sample from distribution but plot mean for simplicity)
    plt.plot(x_target.numpy()[0], dist.loc.detach().numpy()[0], 
             c='b', alpha=0.5)
    plt.xlim(-pi, pi)

plt.show()



from neural_processes import context_target_split

# Extract a batch from data_loader
for batch in data_loader:
    break

# Use batch to create random set of context points
x, y = batch
x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                  25, 
                                                  num_target)

# Create a set of target points corresponding to entire [-pi, pi] range
x_target = torch.Tensor(np.linspace(-pi, pi, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)


for i in range(64):
    # Neural process returns distribution over y_target
    p_y_pred = neuralprocess.test_step((x_context, y_context, x_target), None)
    # Extract mean of distribution
    mu = p_y_pred.loc.detach()
    plt.plot(x_target.numpy()[0], mu.numpy()[0], 
             alpha=0.05, c='b')

# plt.scatter(x_context[0].numpy(), y_context[0].numpy(), c='k')
plt.show()