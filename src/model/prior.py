import torch
import torch.nn as nn
from functools import partial
from .utils_cl import generate_4d_identity_cov
from .linearization import induced_prior_fn_refactored

class CLPrior(nn.Module):
    def __init__(
        self,
        prior_type,
        output_size,
        full_ntk,
        prior_mean,
        prior_cov
    ):
        self.prior_type = prior_type
        self.output_size = output_size
        self.full_ntk = full_ntk
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.stochastic_linearization_prior = False

    def make_prior_fn(
        self,
        model,
        full_ntk,
        prior_type=None,
    ):
        if prior_type is None:
            prior_type = self.prior_type

        if prior_type == 'bnn_induced':
            prior_fn = partial(
                induced_prior_fn_refactored,
                model=model,
                full_ntk=full_ntk
            )

        elif prior_type == 'fixed':
            def prior_fn(context_points, prior_mean=None):
                shape_mean = context_points.shape
                if prior_mean is not None:
                    prior_means = torch.ones(shape_mean).to(prior_mean.device) * prior_mean
                    prior_covs = (generate_4d_identity_cov(*shape_mean) * self.prior_cov).to(prior_mean.device)
                else:
                    prior_means = torch.ones(shape_mean) * self.prior_mean
                    prior_covs = generate_4d_identity_cov(*shape_mean, full_ntk) * self.prior_cov
                return prior_means, prior_covs
        else:
            raise NotImplementedError(prior_type)

        return prior_fn