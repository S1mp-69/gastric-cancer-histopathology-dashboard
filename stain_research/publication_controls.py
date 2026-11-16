"""Prospective controls only. Does not mutate the completed training campaign.

Permutation preserves the finite/valid training strength multiset. Within-class
permutation preserves its class-conditional distribution as an additional
falsification control, not a deployable label-dependent policy.
"""
import numpy as np

def assigned_strengths(alpha, valid, class_codes, seed, epoch, within_class=False):
    alpha=np.asarray(alpha,dtype=float);valid=np.asarray(valid,dtype=bool)
    codes=np.asarray(class_codes)
    if alpha.ndim!=1 or valid.shape!=alpha.shape or codes.shape!=alpha.shape:
        raise ValueError('Aligned one-dimensional training arrays required')
    if not np.isfinite(alpha[valid]).all() or np.any((alpha[valid]<0)|(alpha[valid]>1)):
        raise ValueError('Valid strengths must be finite and in [0,1]')
    result=alpha.copy();rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,710301]))
    groups=[np.flatnonzero(valid & (codes==c)) for c in sorted(set(codes))] if within_class else [np.flatnonzero(valid)]
    for indices in groups:result[indices]=alpha[rng.permutation(indices)]
    # Invalid decompositions remain unchanged-image fallbacks.
    result[~valid]=0
    return result

def component_strengths(features, low, high, component):
    if component not in [0,1]:raise ValueError('0=log condition number, 1=residual')
    scaled=np.clip((np.asarray(features)-low)/np.maximum(np.asarray(high)-low,1e-8),0,1)
    return 1-.9*scaled[:,component]
