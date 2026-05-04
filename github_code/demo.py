"""
Quick demo — run D-PPR on the Karate Club graph.

Usage:
    python demo.py
"""

import networkx as nx
from evaluate import evaluate


def main():
    # ---- Built-in small graph (no external data needed) ----
    G = nx.karate_club_graph()
    evaluate(G, name="Karate Club", alpha_ppr=0.85, alpha_dist=0.05)

    # ---- Uncomment to run on your own edge-list files ----
    # G = nx.read_edgelist("data/airchina_all.txt", nodetype=int, comments="#")
    # evaluate(G, name="Air-China", alpha_ppr=0.85, alpha_dist=0.1)


if __name__ == "__main__":
    main()
