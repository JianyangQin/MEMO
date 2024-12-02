import torch.nn.functional as F
from .linearization import bnn_linearized_predictive
from .utils_cl import _kl_divergence_min_max_dim, _split_mean_cov

def consolidation_loss(
    model,
    prior_fn,
    context_points,
    context_adjs,
    conflict_mapping,
    stable_mapping,
    elbo_weights,
    device,
    full_ntk,
    len_pred=12
):
    params = {k: v.detach() for k, v in model.named_parameters()}
    conflict_kl = 0.
    stable_kl = 0.

    prior_means, prior_covs = prior_fn(context_points=context_points, context_adjs=context_adjs, context_mapping = conflict_mapping + stable_mapping)
    prior_means, prior_covs = prior_means.to(device), prior_covs.to(device)

    n_samples, n_nodes = context_points.shape[0], len(conflict_mapping)+len(stable_mapping)
    min_dim, max_dim = 0, len_pred

    means, covs = bnn_linearized_predictive(
        model,
        params,
        context_points,
        context_adjs,
        conflict_mapping + stable_mapping,
        full_ntk=full_ntk,
        for_loop=False,
        identity_cov=False
    )

    conflict_means, conflict_covs, stable_means, stable_covs = _split_mean_cov(
        means, covs, n_samples, n_nodes, len_pred,
        list(range(len(conflict_mapping))), list(range(len(stable_mapping), len(conflict_mapping+stable_mapping)))
    )
    prior_conflict_means, prior_conflict_covs, prior_stable_means, prior_stable_covs = _split_mean_cov(
        prior_means, prior_covs, n_samples, n_nodes, len_pred,
        list(range(len(conflict_mapping))), list(range(len(stable_mapping), len(conflict_mapping+stable_mapping)))
    )

    # conflict_means, conflict_covs, stable_means, stable_covs = _split_mean_cov(
    #     means, covs, n_samples, n_nodes, n_preds, int(n_nodes // 2)
    # )
    # prior_conflict_means, prior_conflict_covs, prior_stable_means, prior_stable_covs = _split_mean_cov(
    #     prior_means, prior_covs, n_samples, n_nodes, n_preds, int(n_nodes // 2)
    # )

    cm_max, pm_max = conflict_means.max(), prior_conflict_means.max()
    cc_max, pc_max = conflict_covs.max(), prior_conflict_covs.max()
    mean_max = cm_max if cm_max > pm_max else pm_max
    covs_max = cc_max if cc_max > pc_max else pc_max

    conflict_kl += 0.1 * _kl_divergence_min_max_dim(
        conflict_means / mean_max,
        prior_conflict_means / mean_max,
        conflict_covs / covs_max,
        prior_conflict_covs / covs_max,
        min_dim, max_dim, 1e-6, full_cov=full_ntk
    )

    stable_kl += 0.1 * _kl_divergence_min_max_dim(
        stable_means / stable_means.max(),
        prior_stable_means / prior_stable_means.max(),
        stable_covs / stable_covs.max(),
        prior_stable_covs / prior_stable_covs.max(),
        min_dim, max_dim, 1e-6,
        full_cov=full_ntk
    )

    elbo = (1 - elbo_weights) * conflict_kl - elbo_weights * stable_kl

    # elbo = (1 - elbo_weights) * stable_kl - elbo_weights * conflict_kl

    # elbo = conflict_kl + stable_kl

    return elbo

def pred_cls_loss(y_true, y_pred, k_true, k_pred):
    l1 = pred_loss(y_pred, y_true)
    loss = l1

    l2 = pred_loss(k_pred, k_true)
    loss = loss + 50 * l2

    return loss

def pred_loss(y_pred, y_true):
    # if scaler is not None:
    #     y_pred = scaler.inverse_transform(y_pred)
    #     y_true = scaler.inverse_transform(y_true)

    loss = F.mse_loss(y_pred.reshape(-1), y_true.reshape(-1), reduction="mean")
    return loss

def cls_loss(y_pred, y_true):
    _, _, n_clusters = y_true.shape
    y_pred = y_pred.reshape(-1, n_clusters)
    y_true = y_true.reshape(-1, n_clusters)
    loss = F.cross_entropy(y_pred, y_true)
    return loss