import numpy as np
import torch.nn as nn
import torch
import warnings
from contextlib import contextmanager
from .utils import to_saliency_map, get_tqdm, ifnone
from .pytorch import _SpatialGaussianKernel
from .model_zoo import get_module
from .pytorch import _IBAForwardHook


class ImageIBA(nn.Module):

    def __init__(self,
                 img,
                 img_mask,
                 context=None,
                 sigma=1.,
                 img_eps_std=None,
                 img_eps_mean=None,
                 initial_alpha=5.0,
                 feature_mean=None,
                 feature_std=None,
                 progbar=False,
                 reverse_lambda=False,
                 combine_loss=False,
                 device='cuda:0'):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.alpha = None  # Initialized on first forward pass
        self.img_mask = img_mask
        self.img = img
        self.context = context
        self.img_eps_std = img_eps_std
        self.img_eps_mean = img_eps_mean
        self.progbar = progbar
        self.sigmoid = nn.Sigmoid()
        self._buffer_capacity = None  # Filled on forward pass, used for loss
        self.sigma = sigma
        self.device = device
        self._mean = feature_mean
        self._std = feature_std
        self._restrict_flow = False
        self.reverse_lambda = reverse_lambda
        self.combine_loss = combine_loss

        # Attach the bottleneck after the input as forward hook
        # TODO we put this hook at input position of the first model. Make sure to pass this position as parameter
        if self.context is not None:
            self._hook_handle = get_module(
                self.context.classifier,
                'embedding').register_forward_hook(
                    _IBAForwardHook(self, 'output'))
        elif self.layer is not None:
            self._hook_handle = self.layer.register_forward_hook(
                _IBAForwardHook(self, 'output'))
        else:
            raise ValueError(
                'context and layer cannot be None at the same time')

        # initialize alpha
        if self.alpha is None:
            self._build()

    def _reset_alpha(self, sentence_length):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha = nn.Parameter(torch.full(self.img_mask.shape,
                                                 self.initial_alpha,
                                                 device=self.device),
                                      requires_grad=True)
            # self.alpha = nn.Parameter(torch.full(self.alpha.expand(sentence_length, 1, -1).shape,
            #                                      self.initial_alpha,
            #                                      device=self.device),
            #                           requires_grad=True)

    def _build(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        """
        shape = self.img_mask.shape
        self.alpha = nn.Parameter(torch.full(shape,
                                             self.initial_alpha,
                                             device=self.device),
                                  requires_grad=True)
        if self.sigma is not None and self.sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(
                2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            self.smooth = _SpatialGaussianKernel(kernel_size, self.sigma,
                                                 shape[1]).to(self.device)
        else:
            self.smooth = None

    def forward(self, x):
        """
        You don't need to call this method manually.

        The iba acts as a model layer, passing the information in `x` along to the next layer
        either as-is or by restricting the flow of infomration.
        We use it also to estimate the distribution of `x` passing through the layer.
        """
        if self._restrict_flow:
            return self._do_restrict_information(x, self.alpha)
        return x

    @staticmethod
    def _calc_capacity(mu, log_var):
        """ Return the feature-wise KL-divergence of p(z|x) and q(z) """
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    @staticmethod
    def _kl_div(x, g, image_mask, img_eps_mean, img_eps_std, lambda_, mean_x,
                std_x):
        """
        x:
        g:
        img_eps_mean:
        img_eps_std:
        img_mask: mask generated from GAN
        lambda_: learning parameter, img mask
        mean_x:
        std_x:

        """
        mean_x = 0
        std_x = 1
        r_norm = (x - mean_x + image_mask *
                  (mean_x - g)) / ((1 - image_mask * lambda_) * std_x)
        var_z = (1 - lambda_)**2 / (1 - image_mask * lambda_)**2

        log_var_z = torch.log(var_z)

        mu_z = r_norm * lambda_

        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity

    def _do_restrict_information(self, g, alpha):
        """ Selectively remove information from x by applying noise """
        if alpha is None:
            raise RuntimeWarning(
                "Alpha not initialized. Run _init() before using the bottleneck."
            )

        # Smoothen and expand alpha on batch dimension
        lamb = self.sigmoid(alpha)
        lamb = lamb.expand(g.shape[0], 1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # sample from random variable x
        eps = g.data.new(g.size()).normal_()
        ε_img = self.img_eps_std * eps + self.img_eps_mean
        # x = self.img_mask * g + (1-self.img_mask) * eps
        x = g
        self.x = x

        # calculate kl divergence
        self._mean = ifnone(self._mean, torch.tensor(0.).to(self.device))
        self._std = ifnone(self._std, torch.tensor(1.).to(self.device))
        self._buffer_capacity = self._kl_div(x, g, self.img_mask,
                                             self.img_eps_mean,
                                             self.img_eps_std, lamb, self._mean,
                                             self._std)

        # apply mask on sampled x
        eps = x.data.new(x.size()).normal_()
        ε = self._std * eps + self._mean
        λ = lamb
        if self.reverse_lambda:
            #TODO rewrite
            z = λ * ε + (1 - λ) * x
        elif self.combine_loss:
            z_positive = λ * x + (1 - λ) * ε
            z_negative = λ * ε + (1 - λ) * x
            z = torch.cat((z_positive, z_negative))
        else:
            z = λ * x + (1 - λ) * ε

        return z

    @contextmanager
    def restrict_flow(self):
        """
        Context mananger to enable information supression.

        Example:
            To make a prediction, with the information flow being supressed.::

                with iba.restrict_flow():
                    # now noise is added
                    model(x)
        """
        self._restrict_flow = True
        try:
            yield
        finally:
            self._restrict_flow = False

    def analyze(self,
                input_t,
                model_loss_fn,
                mode="saliency",
                beta=10.0,
                opt_steps=10,
                lr=1.0,
                batch_size=10):
        """
        Generates a heatmap for a given sample. Make sure you estimated mean and variance of the
        input distribution.

        Args:
            input_t: input img of shape (1, C, H W)
            model_loss_fn: closure evaluating the model
            mode: how to post-process the resulting map: 'saliency' (default) or 'capacity'
            beta: beta of the combined loss.
            opt_steps: optimization steps.
            lr: learning rate
            batch_size: batch size

        Returns:
            The heatmap of the same shape as the ``input_t``.
        """
        assert input_t.shape[1] == 1, "We can only fit one sample a time"
        batch = batch = input_t.expand(-1, batch_size)

        # Reset from previous run or modifications
        self._reset_alpha(input_t.shape[0])
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        self._loss = []
        self._alpha_grads = []
        self._model_loss = []
        self._information_loss = []

        opt_range = range(opt_steps)
        try:
            tqdm = get_tqdm()
            opt_range = tqdm(opt_range,
                             desc="Training Bottleneck",
                             disable=not self.progbar)
        except ImportError:
            if self.progbar:
                warnings.warn("Cannot load tqdm! Sorry, no progress bar")
                self.progbar = False

        with self.restrict_flow():
            for _ in opt_range:
                optimizer.zero_grad()
                model_loss = model_loss_fn(batch)
                # Taking the mean is equivalent of scaling the sum with 1/K
                information_loss = self.capacity().mean()
                if self.reverse_lambda:
                    loss = -model_loss + beta * information_loss
                else:
                    loss = model_loss + beta * information_loss
                loss.backward(retain_graph=True)
                optimizer.step()

                self._alpha_grads.append(self.alpha.grad.cpu().numpy())
                self._loss.append(loss.item())
                self._model_loss.append(model_loss.item())
                self._information_loss.append(information_loss.item())

        print(self._model_loss)
        print(self._information_loss)
        return self._get_saliency(mode=mode, shape=input_t.shape[2:])

    def capacity(self):
        """
        Returns a tensor with the capacity from the last input, averaged
        over the redundant batch dimension.
        Shape is ``(self.channels, self.height, self.width)``
        """
        return self._buffer_capacity.mean(dim=1)

    def _get_saliency(self, mode='saliency', shape=None):
        capacity_np = self.capacity().detach().cpu().numpy()
        if mode == "saliency":
            # In bits, summed over channels, scaled to input
            # print(np.around(capacity_np.sum(1), decimals=2))
            print(capacity_np.sum(1))
            return capacity_np.sum(1)
        elif mode == "capacity":
            # In bits, not summed, not scaled
            return capacity_np / float(np.log(2))
        else:
            raise ValueError
