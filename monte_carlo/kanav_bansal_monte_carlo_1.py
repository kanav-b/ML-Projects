#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wright-Fisher forward simulation with demography and selection."
    )
    parser.add_argument("config", help="Config CSV (wright_fisher_config.csv)")
    parser.add_argument("demography", help="Demography CSV (wright_fisher_demography.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="forward_results.tsv", help="Output TSV file")
    return parser.parse_args()


def read_config(config_path):
    df = pd.read_csv(config_path)
    params = dict(zip(df["parameter"], df["value"]))

    # Cast to appropriate types
    mutation_rate = float(params["mutation_rate"])
    sequence_length = int(float(params["sequence_length"]))
    total_generations = int(float(params["total_generations"]))
    selection_coefficient = float(params["selection_coefficient"])
    beneficial_mutation_time = int(float(params["beneficial_mutation_time"]))

    return {
        "mutation_rate": mutation_rate,
        "sequence_length": sequence_length,
        "total_generations": total_generations,
        "selection_coefficient": selection_coefficient,
        "beneficial_mutation_time": beneficial_mutation_time,
    }


def build_population_schedule(demo_path, total_generations):
    """
    Demography file format:
    generation,population_size
    0,10000
    1000,100
    1050,10000

    We assume piecewise-constant N between specified change points.
    Returns an array pop_sizes[g] for g=0..total_generations-1.
    """
    demo = pd.read_csv(demo_path)
    demo = demo.sort_values("generation")

    generations = demo["generation"].values
    sizes = demo["population_size"].values

    pop_sizes = np.zeros(total_generations, dtype=int)

    for i in range(len(generations)):
        start_gen = generations[i]
        if i + 1 < len(generations):
            end_gen = generations[i + 1]
        else:
            end_gen = total_generations
        start_gen = max(start_gen, 0)
        end_gen = min(end_gen, total_generations)
        pop_sizes[start_gen:end_gen] = sizes[i]

    # If some generations at the beginning are zero (shouldn't happen with gen=0),
    # fill them with the first specified size.
    if pop_sizes[0] == 0:
        pop_sizes[:generations[0]] = sizes[0]

    return pop_sizes


def compute_diversity(genomes):
    """
    genomes: numpy array shape (N, L) with 0/1 alleles.
    Returns:
        num_mutations: number of sites with at least one derived allele
        pi: nucleotide diversity per site
        theta_w: Watterson's theta per site
    """
    N, L = genomes.shape
    if N < 2:
        return 0, 0.0, 0.0

    # Derived allele counts per site
    derived_counts = genomes.sum(axis=0)

    # Segregating sites: 0 < k < N
    seg_mask = (derived_counts > 0) & (derived_counts < N)
    S = seg_mask.sum()

    # Nucleotide diversity π: average pairwise differences per site
    # For each site: 2 * k * (N - k) / (N * (N - 1))
    k = derived_counts.astype(float)
    per_site_pi = 2.0 * k * (N - k) / (N * (N - 1))
    pi = per_site_pi.sum() / L

    # Watterson's theta: S / (a1 * L)
    a1 = np.sum(1.0 / np.arange(1, N))
    theta_w = (S / a1) / L if a1 > 0 else 0.0

    return int(S), float(pi), float(theta_w)


def simulate_wright_fisher(config, pop_sizes, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mu = config["mutation_rate"]
    L = config["sequence_length"]
    total_generations = config["total_generations"]
    s = config["selection_coefficient"]
    beneficial_time = config["beneficial_mutation_time"]

    # Initial population size
    N0 = pop_sizes[0]

    # genomes: shape (N, L) with all ancestral alleles (0)
    genomes = np.zeros((N0, L), dtype=np.int8)

    # Beneficial locus tracked separately (haploid)
    beneficial = np.zeros(N0, dtype=np.int8)

    results = []

    # Precompute harmonic numbers for all possible N up to max N (for speed)
    max_N = pop_sizes.max()
    harmonic_cache = np.zeros(max_N + 1, dtype=float)
    harmonic_cache[1:] = np.cumsum(1.0 / np.arange(1, max_N + 1))

    for gen in range(total_generations):
        N = pop_sizes[gen]
        # Adjust population size if it changed
        if gen > 0:
            N_prev = genomes.shape[0]
            N_next = N

            # Fitness for parent choice (selection acts on beneficial locus)
            if gen >= beneficial_time:
                fitness = 1.0 + s * beneficial
            else:
                fitness = np.ones(N_prev, dtype=float)

            total_fitness = fitness.sum()
            if total_fitness == 0.0:
                probs = np.ones(N_prev) / N_prev
            else:
                probs = fitness / total_fitness

            # Sample parents for new generation
            parent_indices = np.random.choice(
                N_prev, size=N_next, replace=True, p=probs
            )
            genomes = genomes[parent_indices].copy()
            beneficial = beneficial[parent_indices].copy()

        # Introduce beneficial mutation at specified generation
        if gen == beneficial_time:
            if N > 0:
                idx = np.random.randint(0, N)
                beneficial[idx] = 1

        # Add neutral mutations via Poisson(mu * L * N)
        expected_mut = mu * L * N
        num_mut = np.random.poisson(expected_mut)
        for _ in range(num_mut):
            i = np.random.randint(0, N)
            site = np.random.randint(0, L)
            genomes[i, site] = 1  # infinite-sites approximation, no back mutation

        # Compute diversity statistics
        _, pi, theta_w = compute_diversity(genomes)

        # Number of segregating sites (neutral mutations)
        num_seg_sites, _, _ = compute_diversity(genomes)

        # Beneficial allele frequency
        benef_freq = beneficial.mean() if N > 0 else 0.0

        results.append(
            {
                "generation": gen,
                "population_size": N,
                "num_mutations": num_seg_sites,
                "nucleotide_diversity": pi,
                "theta_watterson": theta_w,
                "beneficial_freq": benef_freq,
            }
        )

    return pd.DataFrame(results)


def main():
    args = parse_args()
    config = read_config(args.config)
    pop_sizes = build_population_schedule(
        args.demography,
        total_generations=config["total_generations"],
    )
    df = simulate_wright_fisher(config, pop_sizes, seed=args.seed)

    # Format columns/rounding for nicer output (optional)
    df["nucleotide_diversity"] = df["nucleotide_diversity"].round(6)
    df["theta_watterson"] = df["theta_watterson"].round(6)
    df["beneficial_freq"] = df["beneficial_freq"].round(6)

    df.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()