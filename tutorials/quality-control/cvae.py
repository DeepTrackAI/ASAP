from typing import Optional, Callable

from deeplay.components import MultiLayerPerceptron
from deeplay.applications import Application
from deeplay.external import Optimizer, Adam

import torch
import torch.nn as nn

class ConditionalVariationalAutoEncoder(Application):
    """Conditional Variational Autoencoder (CVAE) Application.

    This application implements a conditional variational autoencoder (CVAE),
    which extends the standard VAE by conditioning both the encoder and 
    decoder on additional information (condition vector c).

    The encoder maps the input and condition into a latent distribution
    parameterized by mean (mu) and log-variance (log_var). A latent vector z
    is sampled using the reparameterization trick. The decoder reconstructs
    the input from z and the same condition.

    The default structure is as follows:

    Encoder:
        1. Concatenate input x and condition c
        2. MLP(input_size + condition_dim → hidden layers → channels[-1])
        3. Linear → mu
        4. Linear → log_var

    Latent sampling:
        z = mu + std * eps  (reparameterization trick)

    Decoder:
        1. Concatenate z and condition c
        2. Linear(latent_dim + condition_dim → channels[-1])
        3. MLP(channels[-1] → hidden layers → input_size)

    Loss:
        total_loss = reconstruction_loss + beta * KL_divergence

    Parameters
    ----------
    input_size: int
        Dimensionality of the input data.
    condition_dim: int
        Dimensionality of the conditioning vector.
    channels: list[int]
        Hidden layer sizes for encoder/decoder MLP.
    encoder: nn.Module, optional
        Custom encoder module. If None, a default MLP is used.
    decoder: nn.Module, optional
        Custom decoder module. If None, a default MLP is used.
    reconstruction_loss: Callable, optional
        Loss function for reconstruction. (Default: BCELoss with sum reduction)
    latent_dim: int
        Dimensionality of latent space.
    beta: float
        Weight for KL divergence term (β-VAE).
    optimizer: Optimizer, optional
        Optimizer used for training. (Default: Adam)

    Attributes
    ----------
    encoder: nn.Module
        Encoder network mapping (x, c) → latent representation.
    decoder: nn.Module
        Decoder network mapping (z, c) → reconstructed input.
    fc_mu: nn.Linear
        Linear layer producing mean of latent distribution.
    fc_var: nn.Linear
        Linear layer producing log-variance of latent distribution.
    fc_dec: nn.Linear
        Linear layer before decoder.
    latent_dim: int
        Dimensionality of latent space.
    beta: float
        KL divergence weight.
    reconstruction_loss: Callable
        Reconstruction loss function.
    optimizer: Optimizer
        Optimization algorithm.

    Input
    -----
    x: float32
        (batch_size, input_size)
    c: float32
        (batch_size, condition_dim)

    Output
    ------
    y_hat: float32
        Reconstructed input (batch_size, input_size)
    mu: float32
        Mean of latent distribution (batch_size, latent_dim)
    log_var: float32
        Log-variance of latent distribution (batch_size, latent_dim)
    z: float32
        Sampled latent vector (batch_size, latent_dim)

    Evaluation
    ----------
    >>> mu, log_var = self.encode(x, c)
    >>> z = self.reparameterize(mu, log_var)
    >>> y_hat = self.decode(z, c)
    >>> return y_hat

    Examples
    --------
    >>> cvae = ConditionalVariationalAutoEncoder(
    ...     input_size=784,
    ...     condition_dim=10,
    ...     channels=[256, 128],
    ...     latent_dim=20,
    ... ).create()
    >>> cvae
    ConditionalVariationalAutoEncoder(
        (encoder): MultiLayerPerceptron(
            (blocks): LayerList(
            (0): LinearBlock(
                (layer): Linear(in_features=794, out_features=256, bias=True)
                (activation): ReLU()
            )
            (1): LinearBlock(
                (layer): Linear(in_features=256, out_features=128, bias=True)
                (activation): ReLU()
            )
            (2): LinearBlock(
                (layer): Linear(in_features=128, out_features=128, bias=True)
                (activation): Identity()
            )
            )
        )
        (fc_mu): Linear(in_features=128, out_features=20, bias=True)
        (fc_var): Linear(in_features=128, out_features=20, bias=True)
        (fc_dec): Linear(in_features=30, out_features=128, bias=True)
        (decoder): MultiLayerPerceptron(
            (blocks): LayerList(
            (0): LinearBlock(
                (layer): Linear(in_features=128, out_features=128, bias=True)
                (activation): ReLU()
            )
            (1): LinearBlock(
                (layer): Linear(in_features=128, out_features=256, bias=True)
                (activation): ReLU()
            )
            (2): LinearBlock(
                (layer): Linear(in_features=256, out_features=784, bias=True)
                (activation): Identity()
            )
            )
        )
        (reconstruction_loss): BCELoss()
        (train_metrics): MetricCollection,
            prefix=train
        )
        (val_metrics): MetricCollection,
            prefix=val
        )
        (test_metrics): MetricCollection,
            prefix=test
        )
        (optimizer): Adam[Adam](lr=0.0001)
        )

    """

    input_size: int
    condition_dim: int
    channels: list
    latent_dim: int
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    beta: float
    reconstruction_loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        input_size: int,
        condition_dim: int,
        channels: list,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        reconstruction_loss: Optional[Callable] = nn.BCELoss(reduction="sum"),
        latent_dim=int,
        beta=1,
        optimizer=None,
        **kwargs,
    ):
        self.encoder = encoder or self._get_default_encoder(input_size, condition_dim, channels)
        self.fc_mu = nn.Linear(
            channels[-1],
            latent_dim,
        )
        self.fc_var = nn.Linear(
            channels[-1],
            latent_dim,
        )
        self.fc_dec = nn.Linear(
            latent_dim + condition_dim,
            channels[-1],
        )
        self.decoder = decoder or self._get_default_decoder(input_size, channels[::-1])
        self.reconstruction_loss = reconstruction_loss or nn.BCELoss(reduction="sum")
        self.latent_dim = latent_dim
        self.beta = beta

        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=1e-4)

        @self.optimizer.params
        def params(self):
            return self.parameters()

    def _get_default_encoder(self, input_size, condition_dim, channels):
        """Creates the default encoder network.

        This method constructs a default encoder using a multilayer perceptron
        (MLP). The encoder takes the concatenation of input data and condition
        vector as input and maps it to a latent feature representation.

        Parameters
        ----------
        input_size : int
            Dimensionality of the input data.
        condition_dim : int
            Dimensionality of the conditioning vector.
        channels : list[int]
            Hidden layer sizes for the encoder.

        Returns
        -------
        nn.Module
            A multilayer perceptron acting as the encoder.

        Examples
        --------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> encoder = cvae._get_default_encoder(784, 10, [256, 128])
        >>> encoder
        MultiLayerPerceptron(
            (blocks): LayerList(
                (0): LinearBlock(
                (layer): Layer[Linear](in_features=794, out_features=256, bias=True)
                (activation): Layer[ReLU]()
                )
                (1): LinearBlock(
                (layer): Layer[Linear](in_features=256, out_features=128, bias=True)
                (activation): Layer[ReLU]()
                )
                (2): LinearBlock(
                (layer): Layer[Linear](in_features=128, out_features=128, bias=True)
                (activation): Layer[Identity]()
                )
            )
            )
        """

        decoder = MultiLayerPerceptron(
            in_features = input_size + condition_dim,
            hidden_features = channels,
            out_features = channels[-1],
        )
        return decoder

    def _get_default_decoder(self, input_size, channels):
        """Creates the default decoder network.

        This method constructs a default decoder using a multilayer perceptron
        (MLP). The decoder maps latent features back to the original input space.

        Parameters
        ----------
        input_size : int
            Dimensionality of the reconstructed output.
        channels : list[int]
            Hidden layer sizes for the decoder.

        Returns
        -------
        nn.Module
            A multilayer perceptron acting as the decoder.

        Examples
        --------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> decoder = cvae._get_default_decoder(784, [128, 256])
        >>> decoder
        MultiLayerPerceptron(
        (blocks): LayerList(
            (0): LinearBlock(
            (layer): Layer[Linear](in_features=128, out_features=128, bias=True)
            (activation): Layer[ReLU]()
            )
            (1): LinearBlock(
            (layer): Layer[Linear](in_features=128, out_features=256, bias=True)
            (activation): Layer[ReLU]()
            )
            (2): LinearBlock(
            (layer): Layer[Linear](in_features=256, out_features=784, bias=True)
            (activation): Layer[Identity]()
            )
        )
        )
        """

        encoder = MultiLayerPerceptron(
        in_features = channels[0],
        hidden_features = channels,
        out_features = input_size,
    )
        return encoder

    def encode(self, x, c):
        """Encodes input data into latent distribution parameters.

        This method concatenates the input data with the condition vector and
        passes it through the encoder network to produce the mean and log-
        variance of the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, input_size).
        c : torch.Tensor
            Conditioning vector of shape (batch_size, condition_dim).

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent distribution.
        log_var : torch.Tensor
            Log-variance of the latent distribution.

        Examples
        --------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> x, c = torch.randn(10, 784), torch.randn(10, 10)
        >>> mu, log_var = cvae.encode(x, c)
        >>> mu.shape, log_var.shape
        (torch.Size([10, 20]), torch.Size([10, 20]))
        """

        if len(c.shape) == 1:
            c = c.unsqueeze(1)
    
        x = torch.cat([x, c], dim=1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Samples a latent vector using the reparameterization trick.

        This method generates a latent variable z by sampling from a normal
        distribution defined by the given mean and log-variance. It enables
        backpropagation through stochastic sampling.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.
        log_var : torch.Tensor
            Log-variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent vector z.

        Examples
        --------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> mu, log_var = torch.randn(10, 20), torch.randn(10, 20)
        >>> z = cvae.reparameterize(mu, log_var)
        >>> z.shaoe
        >>> torch.Size([10, 20])
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, c):
        """Decodes latent variables into reconstructed input.

        This method concatenates the latent vector with the condition vector,
        transforms it through a linear layer, and passes it through the decoder
        to reconstruct the input.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch_size, latent_dim).
        c : torch.Tensor
            Conditioning vector of shape (batch_size, condition_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed input.

        Examples
        --------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> z, c = torch.randn(10, 20), torch.randn(10, 10)
        >>> x_hat = cvae.decode(z, c)
        >>> x_hat.shape
        torch.Size([10, 784])
        """

        if len(c.shape) == 1:
            c = c.unsqueeze(1)
    
        z = torch.cat([z, c], dim=1)
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x
        
    def train_preprocess(self, batch):
        """Preprocesses a batch of data for training.

        This method prepares the input, target, and condition tensors by ensuring
        they are in the correct format (e.g., channel-first if required).

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y, c).

        Returns
        -------
        tuple
            Preprocessed (x, y, c).

        Examples
        -------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> x, c = torch.randn(10, 784), torch.randn(10, 10)
        >>> y = x
        >>> batch = (x, y, c)
        >>> x_p, y_p, c_p = cvae.train_preprocess(batch)
        >>> x_p.shape, y_p.shape, c_p.shape
        (torch.Size([10, 784]), torch.Size([10, 784]), torch.Size([10, 10]))
        """

        x, y, c = batch
        x = self._maybe_to_channel_first(x)
        y = self._maybe_to_channel_first(y)
        c = self._maybe_to_channel_first(c)
        return x, y, c

    val_preprocess = train_preprocess
    test_preprocess = train_preprocess
    
    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        This method processes a batch of data, computes the forward pass,
        calculates reconstruction and KL divergence losses, and logs them.

        Parameters
        ----------
        batch : tuple
            A batch of training data (x, y, c).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Total loss for optimization.

        Examples
        -------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> cvae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> cvae = cvae.create()
        >>> x, c = torch.rand(10, 784), torch.rand(10, 10)
        >>> y = x
        >>> batch = (x, y, c)
        >>> loss_train = cvae.training_step(batch, _)
        >>> loss_train
        tensor(5442.5708, grad_fn=<AddBackward0>)
        """

        x, y, c = self.train_preprocess(batch)
        y_hat, mu, log_var, z = self(x, c)
        rec_loss, KLD = self.compute_loss(y_hat, y, mu, log_var)
        tot_loss = rec_loss + self.beta * KLD
        loss = {"rec_loss": rec_loss, "KL": KLD, "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss

    def test_step(self, batch, batch_idx):
        """Performs a single test step.

        This method evaluates the model on a test batch and logs the
        reconstruction and KL divergence losses.

        Parameters
        ----------
        batch : tuple
            A batch of test data (x, y, c).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Total test loss.

        Examples
        -------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> cvae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> cvae = cvae.create()
        >>> x, c = torch.rand(10, 784), torch.rand(10, 10)
        >>> y = x
        >>> batch = (x, y, c)
        >>> loss_test = cvae.test_step(batch, _)
        >>> loss_test
        tensor(5440.9023, grad_fn=<AddBackward0>)
        """

        x, y, c  = self.test_preprocess(batch)
        y_hat, mu, log_var, z = self(x, c)
        rec_loss, KLD = self.compute_loss(y_hat, y, mu, log_var)
        tot_loss = rec_loss + self.beta * KLD
        loss = {"rec_loss": rec_loss, "KL": KLD, "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"test_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss
        
    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        This method evaluates the model on a validation batch and logs the
        reconstruction and KL divergence losses.

        Parameters
        ----------
        batch : tuple
            A batch of validation data (x, y, c).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Total validation loss.

        Examples
        -------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> cvae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> cvae = cvae.create()
        >>> x, c = torch.rand(10, 784), torch.rand(10, 10)
        >>> y = x
        >>> batch = (x, y, c)
        >>> loss_val = cvae.validation_step(batch, _)
        >>> loss_val
        tensor(5437.9136, grad_fn=<AddBackward0>)
        """

        x, y, c = self.val_preprocess(batch)
        y_hat, mu, log_var, z = self(x, c)
        rec_loss, KLD = self.compute_loss(y_hat, y, mu, log_var)
        tot_loss = rec_loss + self.beta * KLD
        loss = {"rec_loss": rec_loss, "KL": KLD, "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"val_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss
    
    def compute_loss(self, y_hat, y, mu, log_var):
        """Computes reconstruction and KL divergence losses.

        This method calculates the reconstruction loss between the predicted
        output and the target, as well as the KL divergence between the latent
        distribution and a standard normal prior.

        Parameters
        ----------
        y_hat : torch.Tensor
            Reconstructed output.
        y : torch.Tensor
            Ground truth target.
        mu : torch.Tensor
            Mean of latent distribution.
        log_var : torch.Tensor
            Log-variance of latent distribution.

        Returns
        -------
        tuple
            Reconstruction loss and KL divergence.

        Examples
        --------
        >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> cvae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> cvae = cvae.create()
        >>> y_hat, y = torch.rand(10, 784), torch.rand(10, 784)
        >>> mu, log_var = torch.randn(10, 20), torch.randn(10, 20)
        >>> cvae.compute_loss(y_hat, y, mu, log_var)
        (tensor(7845.0801), tensor(177.7961))
        """

        rec_loss = self.reconstruction_loss(y_hat, y)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return rec_loss, KLD

    def forward(self, x, c):
        """Defines the forward pass of the CVAE.

        This method encodes the input and condition into a latent distribution,
        samples a latent vector, and decodes it to reconstruct the input.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        c : torch.Tensor
            Conditioning vector.

        Returns
        -------
        tuple
            (y_hat, mu, log_var, z)

        Examples
        --------
         >>> cvae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> x, c = torch.randn(10, 784), torch.randn(10, 10)
        >>> y_hat, mu, log_var, z = cvae(x, c)
        >>> y_hat.shape, mu.shape, log_var.shape, z.shape
        (torch.Size([10, 784]),
        torch.Size([10, 20]),
        torch.Size([10, 20]),
        torch.Size([10, 20]))
        """

        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decode(z, c)
        return y_hat, mu, log_var, z