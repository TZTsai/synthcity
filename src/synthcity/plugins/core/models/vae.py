# stdlib
from typing import Any, Callable, List, Optional, Sequence, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, sampler
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.callbacks import Callback, TorchModuleWithValidation
from synthcity.utils.constants import DEVICE

# synthcity relative
from .mlp import MLP


class Encoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_embedding: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        random_state: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
        device: Any = DEVICE,
    ) -> None:
        super(Encoder, self).__init__()
        self.device = device
        self.shared = MLP(
            task_type="regression",
            n_units_in=n_units_in,
            n_units_out=n_units_hidden,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden - 1,
            nonlin=nonlin,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(self.device)

        self.mu_fc = nn.Linear(n_units_hidden, n_units_embedding).to(self.device)
        self.logvar_fc = nn.Linear(n_units_hidden, n_units_embedding).to(self.device)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self, X: Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        data = self._append_optional_cond(X, cond)
        shared = self.shared(data)
        mu = self.mu_fc(shared)
        logvar = self.logvar_fc(shared)
        return mu, logvar

    def _append_optional_cond(
        self, X: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if cond is None:
            return X.float()

        return torch.cat([X, cond], dim=-1).float()


class Decoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_embedding: int,
        n_units_out: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        random_state: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
        device: Any = DEVICE,
    ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.model = MLP(
            task_type="regression",
            n_units_in=n_units_embedding,
            n_units_out=n_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=nonlin_out,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(self.device)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: Tensor, cond: Optional[torch.Tensor] = None) -> Tensor:
        data = self._append_optional_cond(X, cond)
        return self.model(data)

    def _append_optional_cond(
        self, X: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if cond is None:
            return X

        return torch.cat([X, cond], dim=1)


class VAE(TorchModuleWithValidation):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.vae.VAE
        :parts: 1


    Basic VAE implementation.

    Args:
        n_features: int
            Number of features in the dataset
        n_units_embedding: int
            Number of units in the latent space
        batch_size: int
            Training batch size
        n_iter: int
            Number of training iterations
        random_state: int
            Random random_state
        lr: float
            Learning rate
        weight_decay: float:
            Optimizer weight decay
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of units in the hidden layer in the decoder
        decoder_nonlin_out: List
            List of activations layout, as generated by the tabular encoder
        decoder_batch_norm: bool
            Use batchnorm in the decoder
        decoder_dropout: float
            Use dropout in the decoder
        decoder_residual: bool
            Use residual connections in the decoder
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of units in the hidden layer in the encoder
        encoder_batch_norm: bool
            Use batchnorm in the encoder
        encoder_dropout: float
            Use dropout in the encoder
        encoder_residual: bool
            Use residual connections in the encoder
        loss_strategy: str
            - standard: classic VAE loss
            - robust_divergence: Algorithm 1 in "Robust Variational Autoencoder for Tabular Data with β Divergence"
        loss_factor: int
            Parameter for the standard loss
        robust_divergence_beta: int
            Parameter for the robust_divergence loss
        dataloader_sampler:
            Custom sampler used by the dataloader, useful for conditional sampling.
        device:
            CPU/CUDA
        extra_loss_cbks:
            Custom loss callbacks. For example, for conditional loss.
        clipping_value:
            Gradients clipping value. Zero disables the feature
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before early stopping is trigged.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_units_embedding: int,
        n_units_conditional: int = 0,
        batch_size: int = 100,
        n_iter: int = 500,
        random_state: int = 0,
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        # Decoder
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 250,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0,
        decoder_residual: bool = True,
        # Encoder
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        # Loss parameters
        loss_strategy: str = "standard",  # standard, robust_divergence
        loss_factor: int = 2,
        robust_divergence_beta: int = 2,  # used for loss_strategy = robust_divergence
        dataloader_sampler: Optional[sampler.Sampler] = None,
        device: Any = DEVICE,
        extra_loss_cbks: List[Callable] = [],
        clipping_value: int = 1,
        valid_size: float = 0,
        callbacks: Sequence[Callback] = (),
        n_iter_print: int = 10,
        # early stopping
        # n_iter_min: int = 100,
        # patience: int = 20,
    ) -> None:
        super().__init__(
            valid_metric=None,  # validation metric is overriden
            valid_size=valid_size,
            callbacks=callbacks,
        )

        if loss_strategy not in ["standard", "robust_divergence"]:
            raise ValueError(f"invalid loss strategy {loss_strategy}")

        self.device = device
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.loss_factor = loss_factor
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_units_embedding = n_units_embedding
        self.loss_strategy = loss_strategy
        self.robust_divergence_beta = robust_divergence_beta
        self.dataloader_sampler = dataloader_sampler
        self.extra_loss_cbks = extra_loss_cbks
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.n_units_conditional = n_units_conditional
        self.clipping_value = clipping_value
        self.n_iter_print = n_iter_print
        # self.n_iter_min = n_iter_min
        # self.patience = patience

        self.encoder = Encoder(
            n_features + n_units_conditional,
            n_units_embedding,
            n_layers_hidden=encoder_n_layers_hidden,
            n_units_hidden=encoder_n_units_hidden,
            nonlin=encoder_nonlin,
            batch_norm=encoder_batch_norm,
            dropout=encoder_dropout,
            device=device,
        )
        self.decoder = Decoder(
            n_units_embedding + n_units_conditional,
            n_features,
            n_layers_hidden=decoder_n_layers_hidden,
            n_units_hidden=decoder_n_units_hidden,
            nonlin=decoder_nonlin,
            nonlin_out=decoder_nonlin_out,
            batch_norm=decoder_batch_norm,
            dropout=decoder_dropout,
            residual=decoder_residual,
            device=device,
        )

        if decoder_nonlin_out is None:
            decoder_nonlin_out = [("none", n_features)]
        self.decoder_nonlin_out = decoder_nonlin_out

    def fit(
        self,
        X: np.ndarray,
        cond: Optional[np.ndarray] = None,
    ) -> Any:
        Xt = self._check_tensor(X)
        condt: Optional[torch.Tensor] = None

        if self.n_units_conditional > 0:
            if cond is None:
                raise ValueError("Expecting valid conditional for training")
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            if cond.shape[1] != self.n_units_conditional:
                raise ValueError(
                    "Expecting conditional with n_units = {self.n_units_conditional}"
                )
            if cond.shape[0] != X.shape[0]:
                raise ValueError(
                    "Expecting conditional with the same length as the dataset"
                )

            condt = self._check_tensor(cond)

        self._train(Xt, condt)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(self, count: int, cond: Optional[np.ndarray] = None) -> np.ndarray:
        self.decoder.eval()

        steps = count // self.batch_size + 1
        data = []

        condt: Optional[torch.Tensor] = None
        if cond is None and self.n_units_conditional > 0:
            # sample from the original conditional
            if self._original_cond is None:
                raise ValueError("Invalid original conditional. Provide a valid value.")
            cond_idxs = torch.randint(len(self._original_cond), (count,))
            cond = self._original_cond[cond_idxs]

        if cond is not None and len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        if cond is not None and len(cond) != count:
            raise ValueError("cond length must match count")

        if cond is not None:
            condt = self._check_tensor(cond)

        for idx in range(steps):
            mean = torch.zeros(self.batch_size, self.n_units_embedding)
            std = torch.ones(self.batch_size, self.n_units_embedding)
            noise = torch.normal(mean=mean, std=std).to(self.device)

            condt_mb: Optional[torch.Tensor] = None
            if condt is not None:
                condt_mb = condt[
                    idx * self.batch_size : min((idx + 1) * self.batch_size, count)
                ]
                noise = noise[: len(condt_mb)]

            fake = self.decoder(noise, condt_mb)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:count]
        return data

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _train_test_split(self, X: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple:
        if self.dataloader_sampler is not None:
            train_idx, test_idx = self.dataloader_sampler.train_test()
        else:
            total = np.arange(0, len(X))
            np.random.shuffle(total)
            split = int(len(total) * 0.8)
            train_idx, test_idx = total[:split], total[split:]

        X_train, X_val = X[train_idx], X[test_idx]
        cond_train, cond_test = None, None
        if cond is not None:
            cond_train, cond_test = cond[train_idx], cond[test_idx]

        return X_train, X_val, cond_train, cond_test

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(
        self,
        X: Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Any:
        self._original_cond = cond

        # X, X_val, cond, cond_val = self._train_test_split(X, cond)

        X, cond = self._set_val_data(X, cond)
        loader = self._dataloader(X, cond)

        optimizer = Adam(
            self.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

        self.on_fit_begin()

        # best_loss = np.inf
        # best_state_dict = None
        # patience = 0
        for epoch in tqdm(range(self.n_iter)):
            self.on_epoch_begin()

            for id_, data in enumerate(loader):
                cond_mb: Optional[torch.Tensor] = None

                if self.n_units_conditional > 0:
                    X, cond_mb = data
                else:
                    X = data[0]

                mu, logvar = self.encoder(X, cond_mb)
                embedding = self._reparameterize(mu, logvar)

                reconstructed = self.decoder(embedding, cond_mb)
                loss = self._loss_function(
                    reconstructed,
                    X,
                    mu,
                    logvar,
                    cond_mb,
                )
                optimizer.zero_grad()
                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.clipping_value
                    )
                loss.backward()
                optimizer.step()

            self.on_epoch_end()

            if epoch % self.n_iter_print == 0:
                val_loss = self.valid_score
                log.debug(f"[{epoch}/{self.n_iter}] Loss: {val_loss}")
                # if val_loss >= best_loss:
                #     patience += 1
                # else:
                #     best_loss = val_loss
                #     best_state_dict = self.state_dict()
                #     patience = 0

                # if patience >= self.patience and epoch >= self.n_iter_min:
                #     log.debug(f"[{epoch}/{self.n_iter}] Early stopping")
                #     break

        # if best_state_dict is not None:
        #     self.load_state_dict(best_state_dict)

        return self

    def validate(self) -> float:
        X_val, cond_val = self.valid_set  # type: ignore
        mu, logvar = self.encoder(X_val, cond_val)
        embedding = self._reparameterize(mu, logvar)
        reconstructed = self.decoder(embedding, cond_val)
        return (
            self._loss_function(
                reconstructed,
                X_val,
                mu,
                logvar,
                cond_val,
            )
            .detach()
            .item()
        )

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _dataloader(self, X: Tensor, cond: Optional[torch.Tensor] = None) -> DataLoader:
        if cond is None:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, cond)

        return DataLoader(
            dataset,
            sampler=self.dataloader_sampler,
            batch_size=self.batch_size,
            pin_memory=False,
        )

    def _loss_function(
        self,
        reconstructed: Tensor,
        real: Tensor,
        mu: Tensor,
        logvar: Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tensor:
        if self.loss_strategy == "robust_divergence":
            loss = self._loss_function_robust_divergence(
                reconstructed, real, mu, logvar
            )
        else:
            loss = self._loss_function_standard(reconstructed, real, mu, logvar)

        for extra_cbk in self.extra_loss_cbks:
            loss += extra_cbk(real, reconstructed, cond)

        return loss

    def _loss_function_standard(
        self,
        reconstructed: Tensor,
        real: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        step = 0

        loss = []
        for activation, length in self.decoder_nonlin_out:
            step_end = step + length
            # reconstructed is after the activation
            if activation == "softmax":
                discr_loss = nn.NLLLoss(reduction="sum")(
                    torch.log(reconstructed[:, step:step_end] + 1e-8),
                    torch.argmax(real[:, step:step_end], dim=-1),
                )
                loss.append(discr_loss)
            else:
                diff = reconstructed[:, step:step_end] - real[:, step:step_end]
                cont_loss = (50 * diff**2).sum()

                loss.append(cont_loss)
            step = step_end

        if step != reconstructed.size()[1]:
            raise RuntimeError(
                f"Invalid reconstructed features. Expected {step}, got {reconstructed.shape}"
            )

        reconstruction_loss = torch.sum(torch.stack(loss)) / real.shape[0]
        KLD_loss = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())) / real.shape[
            0
        ]

        if torch.isnan(reconstruction_loss):
            raise RuntimeError("NaNs detected in the reconstruction_loss")
        if torch.isnan(KLD_loss):
            raise RuntimeError("NaNs detected in the KLD_loss")

        return reconstruction_loss * self.loss_factor + KLD_loss

    def _loss_function_robust_divergence(
        self,
        reconstructed: Tensor,
        real: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        step = 0

        loss = []
        beta = self.robust_divergence_beta
        std = 0.1
        N = len(real)

        for activation, length in self.decoder_nonlin_out:
            step_end = step + length
            feature_recon = reconstructed[:, step:step_end]
            feature_real = real[:, step:step_end]
            if activation == "softmax":
                # cat_probs = torch.argmax(feature_real, dim=-1)
                # _, cat_probs = torch.unique(cat_probs, return_counts=True)
                # cat_probs = (cat_probs / N) ** (beta + 1)

                # discr_loss = torch.sum((feature_recon ** beta - 1))
                # discr_loss = -(beta + 1) / (beta * N) * discr_loss
                # discr_loss += torch.sum(cat_probs)

                # TODO: debug why robust cross entropy is not working
                discr_loss = nn.NLLLoss(reduction="sum")(
                    torch.log(reconstructed[:, step:step_end] + 1e-8),
                    torch.argmax(real[:, step:step_end], dim=-1),
                )

                loss.append(discr_loss)
            else:
                cont_loss = (-beta / (2 * std**2)) * torch.norm(
                    feature_recon - feature_real, p=2
                ) ** 2
                cont_loss = (1 / (2 * np.pi * std**2) ** (beta / 2)) * torch.exp(
                    cont_loss
                )
                cont_loss = torch.sum(cont_loss)
                cont_loss = (-(beta + 1) / (beta * N)) * cont_loss + (beta + 1) / beta
                cont_loss = torch.sum(cont_loss)

                loss.append(cont_loss)
            step = step_end

        if step != reconstructed.size()[1]:
            raise RuntimeError(
                f"Invalid reconstructed features. Expected {step}, got {reconstructed.shape}"
            )

        reconstruction_loss = torch.sum(torch.stack(loss)) / real.shape[0]

        KLD_loss = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())) / real.shape[
            0
        ]

        return reconstruction_loss * self.loss_factor + KLD_loss
