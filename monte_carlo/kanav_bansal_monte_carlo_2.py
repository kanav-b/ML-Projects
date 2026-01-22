#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coalescent simulation with variable population size."
    )
    parser.add_argument("config", help="Config CSV (coalescent_config.csv)")
    parser.add_argument("demography", help="Demography CSV (coalescent_demography.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="coalescent_results.tsv", help="Output TSV file")
    return parser.parse_args()


def read_config(config_path):
    df = pd.read_csv(config_path)
    params = dict(zip(df["parameter"], df["value"]))

    mutation_rate = float(params["mutation_rate"])
    sequence_length = int(float(params["sequence_length"]))
    sample_size = int(float(params["sample_size"]))
    replicates = int(float(params["replicates"]))

    return {
        "mutation_rate": mutation_rate,
        "sequence_length": sequence_length,
        "sample_size": sample_size,
        "replicates": replicates,
    }


class Demography:
    """
    Piecewise-constant demography in 'time_ago' units.
    CSV format:
    time_ago,population_size
    0,10000
    50,100
    100,10000
    """

    def __init__(self, demography_path):
        df = pd.read_csv(demography_path)
        df = df.sort_values("time_ago")
        self.times = df["time_ago"].values.astype(float)
        self.sizes = df["population_size"].values.astype(float)

    def N_at(self, t):
        """
        Effective population size at time t (time_ago).
        Use population size from the latest time_ago <= t.
        If t is beyond the last time point, use the last size.
        """
        idx = np.searchsorted(self.times, t, side="right") - 1
        if idx < 0:
            idx = 0
        if idx >= len(self.sizes):
            idx = len(self.sizes) - 1
        return self.sizes[idx]

    def next_change_after(self, t):
        """
        Next demography change time after current time t.
        If there is no further change, return None.
        """
        idx = np.searchsorted(self.times, t, side="right")
        if idx >= len(self.times):
            return None
        return self.times[idx]


def simulate_coalescent_tree(sample_size, demography):
    """
    Simulate one coalescent tree with variable N(t) (backwards time in generations).

    Returns:
        times: dict {node_id: time_ago}
        children: dict {node_id: [child1, child2]}
        root: node_id of MRCA
        t_mrca: time_ago to MRCA
    """
    n = sample_size
    # Initial lineages are samples 0..n-1 at time 0
    active = list(range(n))
    times = {i: 0.0 for i in active}
    children = {}

    current_time = 0.0
    next_node_id = n

    while len(active) > 1:
        k = len(active)
        N = demography.N_at(current_time)
        if N <= 0:
            raise ValueError("Population size must be positive in demography.")

        # Coalescent rate for k lineages (haploid vs diploid scaling is arbitrary as long as consistent)
        # Here we use rate = k(k-1)/(2N) per generation.
        rate = k * (k - 1) / (2.0 * N)
        if rate <= 0:
            # This should not happen for k>1 and N>0
            break

        # Exponential waiting time to next coalescent, assuming constant N until next change
        wait = np.random.exponential(1.0 / rate)

        # Time to next demography change
        next_change = demography.next_change_after(current_time)
        if next_change is not None and current_time + wait > next_change:
            # No coalescent in this interval; jump to change point
            current_time = next_change
            continue

        # Coalescent occurs before next change
        current_time += wait

        # Choose two lineages to coalesce
        i, j = np.random.choice(active, size=2, replace=False)
        new_node = next_node_id
        next_node_id += 1

        times[new_node] = current_time
        children[new_node] = [i, j]

        # Update active lineages
        active.remove(i)
        active.remove(j)
        active.append(new_node)

    root = active[0]
    t_mrca = current_time
    return times, children, root, t_mrca


def compute_branch_lengths(times, children, root):
    """
    Given node times and children, compute:
        - parent dict
        - branch_lengths dict child -> length
        - total_tree_length (sum of all branch lengths)
    """
    parent = {}
    branch_lengths = {}
    for p, child_list in children.items():
        for c in child_list:
            parent[c] = p

    # All nodes except root have a parent
    total_length = 0.0
    for node, time in times.items():
        if node == root:
            continue
        p = parent[node]
        length = times[p] - times[node]
        if length < 0:
            # Numerical issues shouldn't really happen, but just in case
            length = 0.0
        branch_lengths[node] = length
        total_length += length

    return parent, branch_lengths, total_length


def build_descendants(children, root, sample_size):
    """
    Compute descendants (sample indices) for each node.
    Returns dict node -> list of sample indices.
    """
    descendants_cache = {}

    def get_desc(node):
        if node in descendants_cache:
            return descendants_cache[node]
        if node < sample_size:
            descendants_cache[node] = [node]
            return descendants_cache[node]
        # Internal node
        desc = []
        for ch in children[node]:
            desc.extend(get_desc(ch))
        descendants_cache[node] = desc
        return desc

    get_desc(root)
    return descendants_cache


def place_mutations(times, children, root, sample_size, L, mu):
    """
    Place mutations on the tree.

    Returns:
        genomes: array shape (sample_size, L) with 0/1 alleles.
        num_mutations: total number of mutations drawn from Poisson
        total_tree_length: total branch length (for reporting)
    """
    parent, branch_lengths, total_tree_length = compute_branch_lengths(times, children, root)
    if total_tree_length <= 0:
        # Degenerate tree
        genomes = np.zeros((sample_size, L), dtype=np.int8)
        return genomes, 0, total_tree_length

    # Draw total number of mutations on the whole tree
    expected_mut = mu * L * total_tree_length
    num_mutations = np.random.poisson(expected_mut)

    # Set up edges (each child node defines an edge from parent -> child)
    edges = list(branch_lengths.keys())
    lengths = np.array([branch_lengths[e] for e in edges], dtype=float)
    cumulative = np.cumsum(lengths)
    tree_length = cumulative[-1]  # should equal total_tree_length

    # Precompute descendants of each node
    descendants = build_descendants(children, root, sample_size)

    # Initialize genomes: all ancestral 0
    genomes = np.zeros((sample_size, L), dtype=np.int8)

    for _ in range(num_mutations):
        # Pick random position along the tree
        u = np.random.random() * tree_length
        idx = np.searchsorted(cumulative, u)
        edge_child = edges[idx]

        # Random site along sequence
        site = np.random.randint(0, L)

        # All descendant leaves of edge_child get derived allele
        leaf_indices = descendants[edge_child]
        genomes[leaf_indices, site] = 1

    return genomes, num_mutations, total_tree_length


def compute_diversity_from_genomes(genomes):
    N, L = genomes.shape
    if N < 2:
        return 0, 0.0, 0.0

    derived_counts = genomes.sum(axis=0)
    # Segregating sites
    seg_mask = (derived_counts > 0) & (derived_counts < N)
    S = seg_mask.sum()

    k = derived_counts.astype(float)
    per_site_pi = 2.0 * k * (N - k) / (N * (N - 1))
    pi = per_site_pi.sum() / L

    a1 = np.sum(1.0 / np.arange(1, N))
    theta_hat = (S / a1) / L if a1 > 0 else 0.0

    return int(S), float(pi), float(theta_hat)


def run_replicates(config, demography, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mu = config["mutation_rate"]
    L = config["sequence_length"]
    n = config["sample_size"]
    R = config["replicates"]

    results = []

    for r in range(1, R + 1):
        # 1) Simulate coalescent tree
        times, children, root, t_mrca = simulate_coalescent_tree(n, demography)

        # 2) Place mutations on branches
        genomes, num_mutations, total_tree_length = place_mutations(
            times, children, root, n, L, mu
        )

        # 3) Compute diversity statistics
        _, pi, theta_hat = compute_diversity_from_genomes(genomes)

        results.append(
            {
                "replicate": r,
                "total_tree_length": total_tree_length,
                "time_to_mrca": t_mrca,
                "num_mutations": num_mutations,
                "theta_estimate": theta_hat,
                "nucleotide_diversity": pi,
            }
        )

    return pd.DataFrame(results)


def main():
    args = parse_args()
    config = read_config(args.config)
    demography = Demography(args.demography)
    df = run_replicates(config, demography, seed=args.seed)

    # Optional rounding for readability
    df["total_tree_length"] = df["total_tree_length"].round(6)
    df["time_to_mrca"] = df["time_to_mrca"].round(6)
    df["theta_estimate"] = df["theta_estimate"].round(6)
    df["nucleotide_diversity"] = df["nucleotide_diversity"].round(6)

    df.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()