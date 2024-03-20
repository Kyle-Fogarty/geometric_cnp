'''
[Neural Process Model]
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

import lightning as pl
from torchtyping import TensorType

import numpy as np

class NeuralProcess(pl.LightningModule):
    '''
    Reimplementation of the Neural Process model using PyTorch Lightning.
    This builds on the pytorch implementation of the Neural Process model
    found in [https://github.com/EmilienDupont/neural-processes].
    '''
    def __init__(self, in_dim       : int,  # Dimension of the input space
                       out_dim      : int,  # Dimension of the output space
                       context_dim  : int,  # Dimension of the context embedding
                       latent_dim   : int,  # Dimension of the embedding -> Normal
                       hidden_dim   : int): # Dimension of the hidden layers in the encoder and decoder
        
        super(NeuralProcess, self).__init__()
        
        # Save the parameters
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.context_dim = context_dim
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim


        # Initialize Neural Process Components

        self.patch_encoder          = Encoder(in_dim, out_dim, hidden_dim, context_dim)
        self.latent_to_distribution = MuSigmaEncoder(context_dim, latent_dim)
        self.target_decoder         = Decoder(in_dim, latent_dim, hidden_dim, out_dim)

        self.encoding_aggregator    = lambda x: torch.mean(x, dim=1) # Defined with a lambda function such that we can
                                                                     # redefine the aggregation function definition from outside of 
                                                                     # this class.

    def training_step(self, batch, batch_idx):
        
        xs, ys = batch
        x_context, y_context, x_target, y_target = context_target_split(xs, ys, 4, 4)

        # (1) Pass the context point through the patch encoder to get the latent space representation
        
        context_encodings = self.patch_encoder(x_context, y_context)
        context_agg_encoding = self.encoding_aggregator(context_encodings)
        context_distribution = self.latent_to_distribution(context_agg_encoding)
        
        # (Training Only) -> Pass the target point and true point through the patch 
        # encoder.

        target_encodings = self.patch_encoder(x_target, y_target)
        target_agg_encoding = self.encoding_aggregator(target_encodings)
        target_distribution = self.latent_to_distribution(target_agg_encoding)

        # (Training Only) -> Sample from the target distribution
        z_samples = target_distribution.sample()

        # (2) Pass the target point and the sample from the latent space through the decoder
        pdf_y_prediction = self.target_decoder(x_target, z_samples) # <- Assumes a N(mu, sigma) distribution

        # (3) Calculate the loss
        loss = self.compute_loss(pdf_y_prediction, y_target, target_distribution, context_distribution)

        return loss
    
    def compute_loss(self, y_pred, y_target, target_distribution, context_distribution):
        log_prob_loss = - y_pred.log_prob(y_target).mean()
        self.log("train_loss | log prob", log_prob_loss)

        kl_div_loss = torch.distributions.kl_divergence(target_distribution, context_distribution).mean()
        self.log("train_loss | distrubtion div", kl_div_loss)

        loss = log_prob_loss + kl_div_loss
        self.log("train_loss | total", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x_context, y_context, x_target, = batch
        # print(x_context.size(), y_context.size())
        context_encodings = self.patch_encoder(x_context, y_context)
        context_agg_encoding = self.encoding_aggregator(context_encodings)
        context_distribution = self.latent_to_distribution(context_agg_encoding)
        
        z_samples = context_distribution.sample()
        
        y_pred_mu, y_pred_sigma = self.target_decoder(x_target, z_samples)
        pdf_y_prediction = Normal(y_pred_mu, y_pred_sigma) # <- Convert mu, sigma to a normal distribution
        
        return pdf_y_prediction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer






class Encoder(nn.Module):
    '''
    
    '''
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        batch, context, _ = x.size()
        input_pairs = torch.cat((x, y), dim=1)
        input_pairs = input_pairs.reshape(-1, self.x_dim + self.y_dim)
        return self.input_to_hidden(input_pairs).reshape(batch,context, self.r_dim)


class MuSigmaEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        batch_size, num_points, _ = x.size()
        # print(batch_size, num_points)
        # print(x.size(), z.size())
    
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target
