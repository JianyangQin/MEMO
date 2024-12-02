import torch
from torch.func import functional_call, vmap, vjp, jvp

def bnn_linearized_predictive(
    model,
    params,
    context_point,
    context_adj,
    context_mapping,
    full_ntk=True,
    for_loop=False,
    identity_cov=False
):
    mean = model(context_point, context_adj, context_mapping).squeeze(0)

    if full_ntk:
        assert not identity_cov, "not implemented"
        cov = empirical_ntk(
            functional_net,
            model,
            params,
            context_point,
            context_point,
            context_adj,
            context_mapping,
            compute='full'
        )
    else:
        if identity_cov:
            cov = torch.ones_like(mean)
        else:
            if for_loop:
                raise NotImplementedError('diag_ntk_for_loop')
            else:
                raise NotImplementedError('neural_tangent_ntk')

    return mean, cov

def induced_prior_fn_refactored(
    model,
    context_points,
    context_adjs,
    context_mapping,
    full_ntk=True,
    identity_cov=False,
    for_loop=False
):

    params = {k: v.detach() for k, v in model.named_parameters()}

    prior_mean, prior_cov = bnn_linearized_predictive(
        model,
        params,
        context_points,
        context_adjs,
        context_mapping,
        full_ntk,
        for_loop,
        identity_cov
    )

    return prior_mean, prior_cov

def functional_net(net, params, x, a, mapping):
    return functional_call(net, params, (x.unsqueeze(0), a, mapping)).squeeze(0)
    # return functional_call(net, params, (x.unsqueeze(0), )).squeeze(0)


def empirical_ntk_ntk_vps(func, net, params, x1, x2, adj, mapping, compute='full'):
    def get_ntk(x1, x2):
        def func_x1(p):
            return func(net, p, x1, adj, mapping)

        def func_x2(p):
            return func(net, p, x2, adj, mapping)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes ``vec @ J(x2).T``
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes ``J(X1) @ vjps``
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice, randomness='different')(basis)

    # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to ``empirical_ntk_ntk_vps`` are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the ``vmaps`` here do.
    result = vmap(vmap(get_ntk, (None, 0), randomness='different'), (0, None), randomness='different')(x1, x2)
    result = result.permute(0, 2, 1, 3)

    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMNM->NM', result)


def empirical_ntk(func, net, params, x1, x2, adj, mapping, compute='full'):
    return empirical_ntk_ntk_vps(func, net, params, x1, x2, adj, mapping, compute)
