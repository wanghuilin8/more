"""
Baseline link prediction methods.

Includes both local heuristics and global/quasi-global methods used for
comparison with D-PPR in the paper.
"""

import numpy as np
import networkx as nx
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import cg
from tqdm import tqdm


# ========================================================================
# Local heuristics
# ========================================================================

def common_neighbors(G, edges):
    """Common Neighbors (CN) index."""
    return [len(list(nx.common_neighbors(G, u, v))) for u, v in edges]


def adamic_adar(G, edges):
    """Adamic–Adar (AA) index."""
    return [s for _, _, s in nx.adamic_adar_index(G, edges)]


def preferential_attachment(G, edges):
    """Preferential Attachment (PA) index."""
    return [s for _, _, s in nx.preferential_attachment(G, edges)]


def resource_allocation(G, edges):
    """Resource Allocation (RA) index."""
    return [s for _, _, s in nx.resource_allocation_index(G, edges)]


# ========================================================================
# Global / quasi-global methods
# ========================================================================

class KatzIndex:
    """Katz index scorer using iterative CG solver.

    Parameters
    ----------
    G : networkx.Graph
    beta : float, default=0.001
        Attenuation factor.  Must be smaller than 1/λ_max(A).
    """

    def __init__(self, G, beta=0.001):
        self.G = G
        self.beta = beta
        self.nodes_list = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}
        n = G.number_of_nodes()
        A = nx.to_scipy_sparse_array(G, nodelist=self.nodes_list, dtype=np.float64)
        self.A_solve = sparse_identity(n) - beta * A
        self._cache = {}

    def predict(self, edges, show_progress=True):
        scores = []
        iterator = tqdm(edges, desc="Katz", disable=not show_progress)
        for u, v in iterator:
            if u not in self.node_to_idx or v not in self.node_to_idx:
                scores.append(0.0)
                continue
            if u not in self._cache:
                e_u = np.zeros(len(self.nodes_list))
                e_u[self.node_to_idx[u]] = 1.0
                vec, _ = cg(self.A_solve.T, e_u, rtol=1e-4, maxiter=100)
                self._cache[u] = vec
            v_idx = self.node_to_idx[v]
            score = self._cache[u][v_idx]
            score -= self.beta * (1.0 if self.G.has_edge(u, v) else 0.0)
            scores.append(score)
        self._cache.clear()
        return scores


class RWR:
    """Random Walk with Restart (RWR) scorer.

    Uses symmetric scoring: score(u,v) = ppr_u[v] + ppr_v[u].

    Parameters
    ----------
    G : networkx.Graph
    restart_prob : float, default=0.15
        Restart probability (1 − damping factor).
    """

    def __init__(self, G, restart_prob=0.15):
        self.G = G
        self.restart_prob = restart_prob
        self.nodes_list = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}
        self.n = G.number_of_nodes()
        self._cache = {}

    def predict(self, edges, show_progress=True):
        # Pre-compute PPR for all unique nodes
        unique = set()
        for u, v in edges:
            unique.add(u)
            unique.add(v)
        unique = [n for n in unique if n in self.node_to_idx]

        for node in tqdm(unique, desc="RWR precompute", disable=not show_progress):
            if node in self._cache:
                continue
            try:
                ppr = nx.pagerank(
                    self.G,
                    alpha=1.0 - self.restart_prob,
                    personalization={node: 1},
                    max_iter=100,
                    tol=1e-6,
                )
                self._cache[node] = np.array([ppr.get(nd, 0) for nd in self.nodes_list])
            except nx.PowerIterationFailedConvergence:
                vec = np.zeros(self.n)
                if node in self.node_to_idx:
                    vec[self.node_to_idx[node]] = 1.0
                self._cache[node] = vec

        scores = []
        for u, v in edges:
            if u not in self.node_to_idx or v not in self.node_to_idx:
                scores.append(0.0)
                continue
            u_idx, v_idx = self.node_to_idx[u], self.node_to_idx[v]
            rwr_u = self._cache.get(u, np.zeros(self.n))
            rwr_v = self._cache.get(v, np.zeros(self.n))
            scores.append(rwr_u[v_idx] + rwr_v[u_idx])
        self._cache.clear()
        return scores
