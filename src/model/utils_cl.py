import torch
import torch.distributions as dist
import tree


def generate_4d_identity_cov(a, b, full_ntk):
    """Generate a 4-dimension identity covariance tensor.

    :param a: the first dimension.
    :param b: the second dimension.
    :return:
        an array of shape (a, b, a, b), where any slice of [:, i, :, i] or
        [j, :, j, :] is an identity matrix.
    """
    # return torch.eye(a)
    eye = torch.eye(a * b)
    stacked = torch.stack(eye.split(b, 0))
    cov = torch.stack(stacked.split(b, 2)).permute(0, 2, 1, 3)
    if full_ntk is False:
        cov = torch.einsum('NMNM->NM', cov)
    return cov


def _slice_cov_diag(cov, index):
    """
    This function slices and takes diagonal

    index is for the output dimension
    """
    ndims = len(cov.shape)
    if ndims == 2:
        cov_i = cov[:, index]
    elif ndims == 3:
        cov_i = cov[:, :, index]
    elif ndims == 4:
        cov_i = cov[:, index, :, index]
    else:
        raise ValueError("Posterior covariance shape not recognized.")
    return cov_i

def kl_full_cov(mean_q, mean_p, cov_q, cov_p):
    """Return KL(q || p).

    :param mean_q: mean of Gaussian distribution q.
    :param mean_p: mean of Gaussian distribution p.
    :param cov_q: covariance of Gaussian distribution q, 2-D tensor.
    :param cov_p: covariance of Gaussian distribution p, 2-D tensor.
    :return:
        KL divergence.
    """
    dims = mean_q.shape[0]
    kl_cov_jitter = torch.tensor(1e-2)  # assuming this constant is defined elsewhere
    _cov_q = cov_q + (torch.eye(dims) * kl_cov_jitter).to(cov_q.device)
    _cov_p = cov_p + (torch.eye(dims) * kl_cov_jitter).to(cov_p.device)

    q = dist.MultivariateNormal(mean_q.t(), covariance_matrix=_cov_q, validate_args=False)
    p = dist.MultivariateNormal(mean_p.t(), covariance_matrix=_cov_p, validate_args=False)
    kl = dist.kl.kl_divergence(q, p)
    return kl

def kl_diag_tfd(mean_q, mean_p, cov_q, cov_p):
    """Return KL(q || p).

    :param mean_q: mean of Gaussian distribution q.
    :param mean_p: mean of Gaussian distribution p.
    :param cov_q: covariance of Gaussian distribution q, 2-D array.
    :param cov_q: covariance of Gaussian distribution p, 2-D array.
    :return:
        KL divergence.
    """
    q = dist.MultivariateNormal(mean_q, torch.diag(torch.sqrt(cov_q)))
    p = dist.MultivariateNormal(mean_p, torch.diag(torch.sqrt(cov_p)))
    return torch.distributions.kl.kl_divergence(q, p)

def _split_mean_cov(mean, cov, n_samples, n_nodes, n_preds,
                    conflict_nodes, stable_nodes):
    mean = mean.reshape(n_samples, n_nodes, -1)

    conflict_mean, stable_mean = mean[:, conflict_nodes], mean[:, stable_nodes]

    conflict_mean = conflict_mean.reshape(-1, n_preds)
    stable_mean = stable_mean.reshape(-1, n_preds)

    if cov.ndim == 2:
        cov = cov.reshape(n_samples, n_nodes, -1)

        conflict_cov, stable_cov = cov[:, conflict_nodes], cov[:, stable_nodes]

        conflict_cov = conflict_cov.reshape(-1, n_preds)
        stable_cov = stable_cov.reshape(-1, n_preds)
    elif cov.ndim == 4:
        cov = cov.reshape(n_samples, n_nodes, n_preds, n_samples, n_nodes, n_preds)

        conflict_cov, stable_cov = cov[:, conflict_nodes][:, :, :, :, conflict_nodes, :], cov[:, stable_nodes][:, :, :, :, stable_nodes, :]

        conflict_cov = conflict_cov.reshape(n_samples * len(conflict_nodes), n_preds, n_samples * len(conflict_nodes), n_preds)
        stable_cov = stable_cov.reshape(n_samples * len(stable_nodes), n_preds, n_samples * len(stable_nodes), n_preds)
    return conflict_mean, conflict_cov, stable_mean, stable_cov

def _kl_divergence_min_max_dim(
    mean_q,
    mean_p,
    cov_q,
    cov_p,
    min_dim,
    max_dim,
    noise=1e-6,
    full_cov=True
):

    kl = 0
    for i in range(min_dim, max_dim):
        mean_q_i = mean_q[:, i]
        mean_p_i = mean_p[:, i]
        cov_q_i = _slice_cov_diag(cov=cov_q, index=i)
        cov_p_i = _slice_cov_diag(cov=cov_p, index=i)
        if full_cov:
            noise_matrix = (torch.eye(cov_q_i.shape[0]) * noise).to(cov_q_i.device)
            cov_q_i += noise_matrix
            cov_p_i += noise_matrix
            kl += kl_full_cov(mean_q_i, mean_p_i, cov_q_i, cov_p_i)
        else:
            if len(cov_q_i.shape) != 1:
                cov_q_i = torch.diag(cov_q_i)
            if len(cov_p_i.shape) != 1:
                cov_p_i = torch.diag(cov_p_i)
            kl += kl_diag_tfd(mean_q_i, mean_p_i, cov_q_i, cov_p_i)
    return kl