"""
D-PPR evaluation utilities.

Provides train/test splitting and a unified evaluation pipeline for
comparing D-PPR against baseline methods.
"""

import random
import time

import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score, roc_auc_score

from dppr import DPPR
from baselines import (
    common_neighbors,
    adamic_adar,
    preferential_attachment,
    resource_allocation,
    KatzIndex,
    RWR,
)


def train_test_split(G, test_ratio=0.1, seed=42):
    """Split a graph into training edges and balanced positive/negative test sets.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (will not be modified).
    test_ratio : float
        Fraction of edges to hold out for testing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    G_train : networkx.Graph
        Training graph with test edges removed.
    test_pos : list of (u, v)
        Positive test edges (real edges held out).
    test_neg : list of (u, v)
        Negative test edges (non-edges sampled uniformly).
    """
    random.seed(seed)
    np.random.seed(seed)

    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    G_train = G.copy()
    edges = list(G.edges())
    random.shuffle(edges)

    num_test = min(int(len(edges) * test_ratio), 20_000)
    test_pos = edges[:num_test]
    G_train.remove_edges_from(test_pos)

    nodes = list(G.nodes())
    test_neg = []
    seen = set()
    while len(test_neg) < num_test:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and (u, v) not in seen:
            test_neg.append((u, v))
            seen.add((u, v))
            seen.add((v, u))

    return G_train, test_pos, test_neg


def evaluate(G, name="Graph", alpha_ppr=0.85, alpha_dist=0.1, test_ratio=0.1, seed=42):
    """Run all seven methods on a graph and print a comparison table.

    Parameters
    ----------
    G : networkx.Graph
    name : str
        Display name for the graph.
    alpha_ppr, alpha_dist : float
        D-PPR hyper-parameters.
    test_ratio : float
        Fraction of edges for testing.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Mapping from method name to (AUPR, AUC-ROC, runtime_seconds).
    """
    print(f"\n{'='*60}")
    print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"{'='*60}")

    G_train, test_pos, test_neg = train_test_split(G, test_ratio, seed)
    test_all = test_pos + test_neg
    labels = [1] * len(test_pos) + [0] * len(test_neg)

    print(f"  Train: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
    print(f"  Test:  {len(test_pos)} pos + {len(test_neg)} neg\n")

    # Build scorers
    katz = KatzIndex(G_train, beta=0.001)
    rwr = RWR(G_train, restart_prob=0.15)
    dppr = DPPR(G_train, alpha_ppr=alpha_ppr, alpha_dist=alpha_dist)

    methods = {
        "CN":    lambda e: common_neighbors(G_train, e),
        "AA":    lambda e: adamic_adar(G_train, e),
        "PA":    lambda e: preferential_attachment(G_train, e),
        "RA":    lambda e: resource_allocation(G_train, e),
        "Katz":  lambda e: katz.predict(e, show_progress=False),
        "RWR":   lambda e: rwr.predict(e, show_progress=False),
        "D-PPR": lambda e: dppr.predict(e, show_progress=False),
    }

    results = {}
    print(f"  {'Method':<8s} {'AUPR':>8s} {'AUC-ROC':>8s} {'Time (s)':>10s}")
    print(f"  {'-'*36}")

    for method_name, scorer in methods.items():
        t0 = time.perf_counter()
        scores = scorer(test_all)
        elapsed = time.perf_counter() - t0

        aupr = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)
        results[method_name] = (aupr, auc, elapsed)

        print(f"  {method_name:<8s} {aupr:8.4f} {auc:8.4f} {elapsed:10.2f}")

    dppr.clear_cache()
    return results
