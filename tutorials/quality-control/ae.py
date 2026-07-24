"""Autoencoder (AE) Application."""

from __future__ import annotations

from typing import Any, Callable

from deeplay.components import MultiLayerPerceptron
from deeplay.applications import Application
from deeplay.external import Optimizer, Adam
import torch
from torch import nn


class AutoEncoder(Application):
    """Autoencoder (AE) Application.

    This application implements an autoencoder (AE).

    The encoder maps the input into a latent space `z`. The decoder
    reconstructs the input from `z`.

    The default structure is as follows:

    Encoder:
        1. MLP(input_size -> hidden layers -> channels[-1])
        2. Linear(channels[-1] -> latent_dim `z`)

    Decoder:
        1. Linear(latent_dim `z` -> channels[-1])
        3. MLP(channels[-1] → hidden layers -> input_size)

    Loss:
        `total_loss = reconstruction_loss`

    Parameters
    ----------
    input_size: int
        Dimensionality of the input data.
    channels: list[int]
        Hidden layer sizes for encoder/decoder MLP.
    encoder: nn.Module | None, optional
        Custom encoder module. If `None` (default), defaults to an MLP.
    decoder: nn.Module | None, optional
        Custom decoder module. If `None` (default), defaults to an MLP.
    reconstruction_loss: Callable, optional
        Loss function for reconstruction.
        Defaults to BCELoss with sum reduction.
    latent_dim: int, optional
        Dimensionality of latent space. Defaults to `2`.
    optimizer: Optimizer | None, optional
        Optimizer used for training. If `None` (default), defaults to Adam.

    Attributes
    ----------
    encoder: nn.Module
        Encoder network mapping (x, ) -> latent representation.
    decoder: nn.Module
        Decoder network mapping (z, ) -> reconstructed input.
    fc_enc : nn.Linear
        Linear layer after encoder.
    fc_dec: nn.Linear
        Linear layer before decoder.
    latent_dim: int
        Dimensionality of latent space.
    reconstruction_loss: Callable
        Reconstruction loss function.
    optimizer: Optimizer
        Optimization algorithm.

    Input
    -----
    x: float32
        (batch_size, input_size)

    Output
    ------
    y_hat: float32
        Reconstructed input (batch_size, input_size)
    z: float32
        Sampled latent vector (batch_size, latent_dim)

    Evaluation
    ----------
    >>> z = self.encode(x)
    >>> y_hat = self.decode(z)
    >>> return y_hat

    Examples
    --------
    >>> ae = AutoEncoder(
    ...     input_size=784,
    ...     channels=[256, 128],
    ...     latent_dim=20,
    ... ).create()
    >>> ae
    AutoEncoder(
        (encoder): MultiLayerPerceptron(
            (blocks): LayerList(
            (0): LinearBlock(
                (layer): Linear(in_features=784, out_features=256, bias=True)
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
        (fc_enc): Linear(in_features=128, out_features=20, bias=True)
        (fc_dec): Linear(in_features=20, out_features=128, bias=True)
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
    channels: list
    latent_dim: int
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    reconstruction_loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        input_size: int,
        channels: list[int],
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        reconstruction_loss: Callable = nn.BCELoss(reduction="sum"),
        latent_dim: int = 2,
        optimizer: Optimizer | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Autoencoder (AE).

        This constructor builds a standard autoencoder that learns a compact
        latent representation of the input data and reconstructs the original
        input from this representation. The model consists of an encoder that
        maps the input to a latent space and a decoder that reconstructs the
        input from the latent representation.

        If no encoder or decoder is provided, default MLP-based architectures
        are constructed using the specified hidden channel sizes.

        Parameters
        ----------
        input_size : int
            Dimensionality of the input data.
        channels : list[int]
            Hidden layer sizes used for encoder and decoder MLPs.
        encoder : nn.Module or None, optional
            Custom encoder network. If None, a default encoder is used.
        decoder : nn.Module or None, optional
            Custom decoder network. If None, a default decoder is used.
        reconstruction_loss : Callable, optional
            Reconstruction loss function. Default is BCELoss with sum reduction.
            Used to measure similarity between reconstructed and target inputs.
        latent_dim : int, optional
            Dimensionality of the latent space. Default is 2.
        optimizer : Optimizer or None, optional
            Optimizer used for training. If None, Adam with lr=1e-4 is used.
        **kwargs : Any
            Additional arguments passed to the parent `Application` class.

        Attributes
        ----------
        encoder : nn.Module
            Encoder network mapping (x, ) → hidden representation.
        decoder : nn.Module
            Decoder network mapping (z, ) → reconstructed input.
        fc_enc : nn.Linear
            Linear projection from encoder output space to latent space.
        fc_dec : nn.Linear
            Linear projection from latent space to decoder input space.
        latent_dim : int
            Dimensionality of latent space.
        reconstruction_loss : Callable
            Reconstruction loss function.
        optimizer : Optimizer
            Optimizer used for training.

        """

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self._get_default_encoder(
                input_size,
                channels,
            )

        self.fc_enc = nn.Linear(
            channels[-1],
            latent_dim,
        )
        self.fc_dec = nn.Linear(
            latent_dim,
            channels[-1],
        )

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = self._get_default_decoder(
                input_size,
                channels[::-1],
            )

        if reconstruction_loss is not None:
            self.reconstruction_loss = reconstruction_loss
        else:
            self.reconstruction_loss = nn.BCELoss(reduction="sum")

        self.latent_dim = latent_dim

        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=1e-4)

        @self.optimizer.params
        def params(self):
            return self.parameters()

    def _get_default_encoder(
        self: AutoEncoder,
        input_size: int,
        channels: list[int],
    ) -> nn.Module:
        """Create the default encoder network.

        This method constructs a default encoder using a multilayer perceptron
        (MLP). The encoder maps the input data to a compact latent
        representation.

        Parameters
        ----------
        input_size: int
            Dimensionality of the input data.
        channels: list[int]
            Hidden layer sizes for the encoder.

        Returns
        -------
        nn.Module
            A multilayer perceptron acting as the encoder.

        Examples
        --------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> encoder = ae._get_default_encoder(784, [256, 128])
        >>> encoder
        MultiLayerPerceptron(
            (blocks): LayerList(
                (0): LinearBlock(
                (layer): Layer[Linear](in_features=784, out_features=256, ...)
                (activation): Layer[ReLU]()
                )
                (1): LinearBlock(
                (layer): Layer[Linear](in_features=256, out_features=128, ...)
                (activation): Layer[ReLU]()
                )
                (2): LinearBlock(
                (layer): Layer[Linear](in_features=128, out_features=128, ...)
                (activation): Layer[Identity]()
                )
            )
            )

        """

        decoder = MultiLayerPerceptron(
            in_features=input_size,
            hidden_features=channels,
            out_features=channels[-1],
        )
        return decoder

    def _get_default_decoder(
        self: AutoEncoder,
        input_size: int,
        channels: list[int],
    ) -> nn.Module:
        """Create the default decoder network.

        This method constructs a default decoder using a multilayer perceptron
        (MLP). The decoder reconstructs the original input from the latent
        representation.

        Parameters
        ----------
        input_size: int
            Dimensionality of the reconstructed output.
        channels: list[int]
            Hidden layer sizes for the decoder.

        Returns
        -------
        nn.Module
            A multilayer perceptron acting as the decoder.

        Examples
        --------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> decoder = ae._get_default_decoder(784, [128, 256])
        >>> decoder
        MultiLayerPerceptron(
            (blocks): LayerList(
                (0): LinearBlock(
                (layer): Layer[Linear](in_features=128, out_features=128, ...)
                (activation): Layer[ReLU]()
                )
                (1): LinearBlock(
                (layer): Layer[Linear](in_features=128, out_features=256, ...)
                (activation): Layer[ReLU]()
                )
                (2): LinearBlock(
                (layer): Layer[Linear](in_features=256, out_features=784, ...)
                (activation): Layer[Identity]()
                )
            )
        )

        """

        encoder = MultiLayerPerceptron(
            in_features=channels[0],
            hidden_features=channels,
            out_features=input_size,
        )

        return encoder

    def encode(
        self: AutoEncoder,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes input data into latent distribution parameters.

        This method first encodes the input data into a hidden representation
        using the encoder network and then projects it into the latent space
        through a linear layer.

        Parameters
        ----------
        x: torch.Tensor
            Input data of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Latent vector `z`.

        Examples
        --------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     condition_dim=10,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> x = torch.randn(10, 784)
        >>> z = ae.encode(x, c)
        >>> z.shape
        torch.Size([10, 20])

        """

        x = self.encoder(x)
        z = self.fc_enc(x)

        return z

    def decode(
        self: AutoEncoder,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent variables into reconstructed input.

        This method passes the latent representation through a linear projection
        layer and the decoder network to reconstruct the original input.

        Parameters
        ----------
        z: torch.Tensor
            Latent vector of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed input.

        Examples
        --------
        >>> ae = ConditionalVariationalAutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> z = torch.randn(10, 20)
        >>> x_hat = ae.decode(z)
        >>> x_hat.shape
        torch.Size([10, 784])

        """

        x = self.fc_dec(z)
        x = self.decoder(x)

        return x

    def train_preprocess(
        self: AutoEncoder,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocesses a batch of data for training.

        This method prepares the input, target  tensors by
        ensuring they are in the correct format (e.g., channel-first if
        required).

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor]
            A tuple containing `(x, y)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Preprocessed (x, y).

        Examples
        -------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> x = torch.randn(10, 784)
        >>> y = x
        >>> batch = (x, y)
        >>> x_p, y_p = ae.train_preprocess(batch)
        >>> x_p.shape, y_p.shape
        (torch.Size([10, 784]), torch.Size([10, 784]))

        """

        x, y = batch

        x = self._maybe_to_channel_first(x)
        y = self._maybe_to_channel_first(y)

        return x, y

    val_preprocess = train_preprocess
    test_preprocess = train_preprocess

    def training_step(
        self: AutoEncoder,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step, the hook for Lightning.

        This method processes a batch of data, computes the forward pass,
        calculates reconstruction loss, and logs it.

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor]
            A batch of training data (x, y).
        batch_idx: int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Total loss for optimization.

        Examples
        -------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> ae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> ae = cvae.create()
        >>> x = torch.rand(10, 784)
        >>> y = x
        >>> batch = (x, y)
        >>> loss_train = ae.training_step(batch, _)
        >>> loss_train
        tensor(5442.5708, grad_fn=<AddBackward0>)

        """

        x, y = self.train_preprocess(batch)
        y_hat, z = self(x)
        tot_loss = self.compute_loss(y_hat, y)

        loss = {"total_loss": tot_loss}
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

    def test_step(
        self: AutoEncoder,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single test step, the hook for Lightning.

        This method evaluates the model on a test batch and logs the
        reconstruction loss.

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor]
            A batch of test data (x, y).
        batch_idx: int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Total test loss.

        Examples
        -------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> ae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> ae = ae.create()
        >>> x = torch.rand(10, 784)
        >>> y = x
        >>> batch = (x, y)
        >>> loss_test = ae.test_step(batch, _)
        >>> loss_test
        tensor(5440.9023, grad_fn=<AddBackward0>)

        """

        x, y = self.test_preprocess(batch)
        y_hat, z = self(x)
        tot_loss = self.compute_loss(y_hat, y)

        loss = {"total_loss": tot_loss}
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

    def validation_step(
        self: AutoEncoder,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Performs a single validation step, the hook for Lightning.

        This method evaluates the model on a validation batch and logs the
        reconstruction loss.

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor]
            A batch of validation data (x, y).
        batch_idx: int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Total validation loss.

        Examples
        -------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> ae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> ae = ae.create()
        >>> x = torch.rand(10, 784)
        >>> y = x
        >>> batch = (x, y)
        >>> loss_val = ae.validation_step(batch, _)
        >>> loss_val
        tensor(5437.9136, grad_fn=<AddBackward0>)

        """

        x, y = self.val_preprocess(batch)
        y_hat, z = self(x)
        tot_loss = self.compute_loss(y_hat, y)

        loss = {"total_loss": tot_loss}
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

    def compute_loss(
        self: AutoEncoder,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Computes reconstruction and KL divergence losses.

        This method calculates the reconstruction loss between the reconstructed
        output and the target input.

        Parameters
        ----------
        y_hat: torch.Tensor
            Reconstructed output.
        y: torch.Tensor
            Ground truth target.

        Returns
        -------
        torch.Tensor
            Reconstruction loss.

        Examples
        --------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... )
        >>> ae.decoder.blocks[2].activated(torch.nn.Sigmoid)
        >>> ae = ae.create()
        >>> y_hat, y = torch.rand(10, 784), torch.rand(10, 784)
        >>> ae.compute_loss(y_hat, y)
        tensor(7845.0801)

        """

        rec_loss = self.reconstruction_loss(y_hat, y)

        return rec_loss

    def forward(
        self: AutoEncoder,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Define the forward pass of the CVAE.

        This method encodes the input and condition into a latent distribution,
        and decodes it to reconstruct the input.

        Parameters
        ----------
        x: torch.Tensor
            Input data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple of PyTorch tensors, `(y_hat, z)`.

        Examples
        --------
        >>> ae = AutoEncoder(
        ...     input_size=784,
        ...     channels=[256, 128],
        ...     latent_dim=20,
        ... ).create()
        >>> x = torch.randn(10, 784)
        >>> y_hat, z = ae(x)
        >>> y_hat.shape, z.shape
        (torch.Size([10, 784]),
         torch.Size([10, 20]))

        """

        z = self.encode(x)
        y_hat = self.decode(z)

        return y_hat, z
