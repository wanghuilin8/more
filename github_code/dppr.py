"""
D-PPR: Diffusion Distance with Personalized PageRank for Link Prediction
=========================================================================

A physics-inspired link prediction framework that combines Personalized
PageRank (PPR) signals with graph diffusion distance.

Reference:
    Wang H., Zhang W., Deng W. "Diffusion Signals Reveal Hidden Connections:
    A Physics-Inspired Framework for Link Prediction via Personalized PageRank
    Signals." Physica A: Statistical Mechanics and its Applications, 2026.
"""

import numpy as np
import networkx as nx
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import cg
from tqdm import tqdm


class DPPR:
    """D-PPR link prediction scorer.

    The D-PPR framework computes link scores through two steps:
      1. Construct a Personalized PageRank (PPR) signal vector for each node,
         capturing multi-scale structural proximity.
      2. Measure the diffusion distance between PPR signal pairs via the
         regularised graph Laplacian  (I + α_dist · L), acting as a spectral
         low-pass filter that smooths out high-frequency noise.

    The link score between nodes u and v is defined as:
        score(u, v) = 1 / (||diffused(s_u - s_v)||_2 + ε)

    Parameters
    ----------
    G : networkx.Graph
        The (training) graph.  Must be undirected and simple.
    alpha_ppr : float, default=0.85
        PPR teleportation parameter (damping factor).
        Higher values explore farther from the source node.
    alpha_dist : float, default=0.1
        Diffusion regularisation strength.
        Controls how strongly the Laplacian smooths the PPR difference signal.
    """

    def __init__(self, G, alpha_ppr=0.85, alpha_dist=0.1):
        self.G = G
        self.alpha_ppr = alpha_ppr
        self.alpha_dist = alpha_dist

        self.nodes_list = list(G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes_list)}
        self.n = G.number_of_nodes()

        # Build the diffusion operator: (I + α_dist · L)
        L = nx.laplacian_matrix(G, nodelist=self.nodes_list).astype(np.float64)
        self.A_solve = sparse_identity(self.n) + self.alpha_dist * L

        self._ppr_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ppr(self, node):
        """Return the PPR signal vector for *node* as a dense array."""
        if node in self._ppr_cache:
            return self._ppr_cache[node]
        try:
            ppr = nx.pagerank(
                self.G,
                alpha=self.alpha_ppr,
                personalization={node: 1},
                max_iter=100,
                tol=1e-6,
            )
            vec = np.array([ppr.get(nd, 0.0) for nd in self.nodes_list])
        except nx.PowerIterationFailedConvergence:
            vec = np.zeros(self.n)
            if node in self.node_to_idx:
                vec[self.node_to_idx[node]] = 1.0
        self._ppr_cache[node] = vec
        return vec

    def _diffusion_distance(self, s_u, s_v):
        """Compute the diffusion distance between two PPR signal vectors."""
        s_diff = s_u - s_v
        if not np.any(s_diff):
            return 0.0
        diffused, exit_code = cg(self.A_solve, s_diff, rtol=1e-6, maxiter=200)
        if exit_code != 0:
            return float("inf")
        return float(np.linalg.norm(diffused))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, edges, show_progress=True):
        """Compute D-PPR link scores for a list of node pairs.

        Parameters
        ----------
        edges : list of (u, v) tuples
            Node pairs to score.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        scores : list of float
            Link prediction scores (higher ⇒ more likely linked).
        """
        # Pre-compute PPR vectors for all unique nodes
        unique_nodes = set()
        for u, v in edges:
            unique_nodes.add(u)
            unique_nodes.add(v)
        unique_nodes = [n for n in unique_nodes if n in self.node_to_idx]

        iterator = tqdm(unique_nodes, desc="PPR precompute", disable=not show_progress)
        for node in iterator:
            self._compute_ppr(node)

        # Score each edge
        scores = []
        for u, v in edges:
            if u not in self.node_to_idx or v not in self.node_to_idx:
                scores.append(0.0)
                continue
            s_u = self._ppr_cache.get(u, np.zeros(self.n))
            s_v = self._ppr_cache.get(v, np.zeros(self.n))
            dist = self._diffusion_distance(s_u, s_v)
            scores.append(1.0 / (dist + 1e-8))
        return scores

    def clear_cache(self):
        """Free cached PPR vectors."""
        self._ppr_cache.clear()
