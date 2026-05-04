# D-PPR: Diffusion Distance with Personalized PageRank for Link Prediction

A physics-inspired link prediction framework that combines **Personalized PageRank (PPR) signals** with **graph diffusion distance**.

> **Paper:** Wang H., Zhang W., Deng W. "Diffusion Signals Reveal Hidden Connections: A Physics-Inspired Framework for Link Prediction via Personalized PageRank Signals." *Physica A: Statistical Mechanics and its Applications*, 2026.

## Method Overview

D-PPR scores the likelihood of a link between nodes *u* and *v* in two steps:

1. **PPR Signal Construction** — Compute a Personalized PageRank vector for each node, capturing multi-scale structural proximity.
2. **Diffusion Distance** — Measure the distance between PPR signal pairs via the regularised graph Laplacian *(I + α · L)*, which acts as a spectral low-pass filter that smooths high-frequency noise.

The final link score is:

$$\text{score}(u, v) = \frac{1}{\| (I + \alpha_{\text{dist}} \cdot L)^{-1} (\mathbf{s}_u - \mathbf{s}_v) \|_2 + \varepsilon}$$

where **s**_u is the PPR vector rooted at node *u*.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo on Karate Club graph
python demo.py
```

**Expected output:**

```
============================================================
  Karate Club: 34 nodes, 78 edges
============================================================
  Train: 34 nodes, 71 edges
  Test:  7 pos + 7 neg

  Method     AUPR  AUC-ROC   Time (s)
  ------------------------------------
  CN         0.87     0.87       0.00
  AA         0.89     0.87       0.00
  PA         0.50     0.50       0.00
  RA         0.90     0.87       0.00
  Katz       0.88     0.83       0.00
  RWR        0.91     0.90       0.01
  D-PPR      0.96     0.93       0.01
```

## Project Structure

```
├── dppr.py            # Core D-PPR algorithm
├── baselines.py       # Baseline methods (CN, AA, PA, RA, Katz, RWR)
├── evaluate.py        # Train/test split and evaluation pipeline
├── demo.py            # Quick demo script
└── requirements.txt   # Python dependencies
```

## API Usage

```python
import networkx as nx
from dppr import DPPR

G = nx.karate_club_graph()

model = DPPR(G, alpha_ppr=0.85, alpha_dist=0.1)
scores = model.predict([(0, 1), (0, 33), (2, 15)])
print(scores)  # Higher score = more likely linked
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_ppr` | 0.85 | PPR damping factor. Higher values explore farther from the source node. |
| `alpha_dist` | 0.1 | Diffusion regularisation strength. Controls spectral smoothing of the PPR difference signal. |

## Citation

If you find this code useful, please cite:

```bibtex
@article{wang2026dppr,
  title   = {Diffusion Signals Reveal Hidden Connections: A Physics-Inspired
             Framework for Link Prediction via Personalized PageRank Signals},
  author  = {Wang, Huilin and Zhang, Wenjun and Deng, Weibing},
  journal = {Physica A: Statistical Mechanics and its Applications},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
