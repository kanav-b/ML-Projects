#!/usr/bin/env python3
"""
Genomic Region Overlap Analysis and Permutation Testing

CLI:
    python firstname_lastname_permutation_test.py <setA_bed> <setB_bed> <genome_fai> <output_dir> [num_permutations]

Environment:
    Works with the provided permutation_test.yaml (numpy, pandas, stdlib only)

Outputs:
    - results.tsv (summary stats)
    - results_per_region.tsv (per-SetB-region p-values and significance)

Notes:
    * Counts base-pair overlaps (weighted), not region counts.
    * Does NOT merge overlapping intervals within a set; each interval is independent.
    * Uses a custom interval tree per chromosome for efficient overlap queries.
    * Permutations preserve per-interval length and chromosome assignment; bounds-safe.
    * Per-region p-values calculated with Bonferroni correction.
"""

from __future__ import annotations
import sys
import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# -----------------------------
# I/O and utility helpers
# -----------------------------

def read_bed(path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Read a 3-column BED (chrom, start, end) into dict: chrom -> list[(start, end)].
    Coordinates are 0-based, end-exclusive, as given.
    Lines starting with '#' are ignored.
    """
    chrom_to_intervals: Dict[str, List[Tuple[int, int]]] = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip().split('\t')
            if len(parts) < 3:
                continue
            chrom, start_s, end_s = parts[0], parts[1], parts[2]
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if end <= start:
                continue
            chrom_to_intervals.setdefault(chrom, []).append((start, end))
    # No merging, but we sort by start for deterministic behavior
    for c in chrom_to_intervals:
        chrom_to_intervals[c].sort()
    return chrom_to_intervals


def read_fai_lengths(path: str) -> Dict[str, int]:
    """Read genome .fai file and return dict: chrom -> length."""
    lengths: Dict[str, int] = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip().split('\t')
            if len(parts) < 2:
                continue
            chrom = parts[0]
            try:
                length = int(parts[1])
            except ValueError:
                continue
            lengths[chrom] = length
    return lengths


def total_bases(intervals: Dict[str, List[Tuple[int, int]]]) -> int:
    return sum(end - start for chrom in intervals for start, end in intervals[chrom])


# -----------------------------
# Interval tree for efficient queries
# -----------------------------

@dataclass
class Interval:
    start: int
    end: int
    idx: Optional[int] = None  # index into Set B (for per-region accounting)


class IntervalTreeNode:
    __slots__ = ("center", "by_start", "by_end", "left", "right")

    def __init__(self, intervals: List[Interval]):
        if not intervals:
            raise ValueError("Cannot build IntervalTreeNode with empty intervals")
        # Choose center as median of interval midpoints to balance the tree
        mids = [ (iv.start + iv.end) // 2 for iv in intervals ]
        self.center = mids[len(mids)//2]

        center_list: List[Interval] = []
        left_list: List[Interval] = []
        right_list: List[Interval] = []
        c = self.center
        for iv in intervals:
            if iv.end <= c:
                left_list.append(iv)
            elif iv.start > c:
                right_list.append(iv)
            else:
                center_list.append(iv)  # intervals crossing the center

        # Store center intervals sorted by start and end for efficient partial scans
        self.by_start = sorted(center_list, key=lambda x: x.start)
        self.by_end = sorted(center_list, key=lambda x: x.end)

        self.left = IntervalTreeNode(left_list) if left_list else None
        self.right = IntervalTreeNode(right_list) if right_list else None

    def query(self, qs: int, qe: int, out: List[Interval]):
        """Append intervals overlapping [qs, qe) into out list."""
        c = self.center
        # Intervals that cross the center may overlap; we can prune using sorted lists
        if self.by_start:
            if qe <= c:
                # Only intervals with start < qe can overlap (all have end > c >= qe)
                for iv in self.by_start:
                    if iv.start >= qe:
                        break
                    # since end > c >= qe > qs is not guaranteed, check overlap
                    if iv.end > qs and iv.start < qe:
                        out.append(iv)
            elif qs > c:
                # Only intervals with end > qs can overlap (all have start <= c < qs?) not guaranteed
                # Scan by_end from first with end > qs
                # by_end is ascending; find first index where end > qs
                lo, hi = 0, len(self.by_end)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if self.by_end[mid].end <= qs:
                        lo = mid + 1
                    else:
                        hi = mid
                for i in range(lo, len(self.by_end)):
                    iv = self.by_end[i]
                    if iv.start < qe and iv.end > qs:
                        out.append(iv)
            else:
                # Query spans the center; all center intervals cross the center, but still verify overlap bounds
                for iv in self.by_start:
                    if iv.end > qs and iv.start < qe:
                        out.append(iv)

        # Recurse to children if query range extends left/right of center
        if self.left and qs < c:
            self.left.query(qs, min(qe, c), out)
        if self.right and qe > c:
            self.right.query(max(qs, c), qe, out)


class IntervalTree:
    def __init__(self, intervals: List[Interval]):
        self.root: Optional[IntervalTreeNode] = IntervalTreeNode(intervals) if intervals else None

    def query(self, qs: int, qe: int) -> List[Interval]:
        if not self.root:
            return []
        out: List[Interval] = []
        self.root.query(qs, qe, out)
        return out


# -----------------------------
# Overlap calculations
# -----------------------------

def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    x = min(a_end, b_end) - max(a_start, b_start)
    return x if x > 0 else 0


def build_B_structures(setB: Dict[str, List[Tuple[int,int]]]):
    """Build per-chromosome interval trees for Set B and keep index mapping.
    Returns:
        B_index: List[Tuple[str, int, int]]  # global index -> (chrom, start, end)
        chrom_to_tree: Dict[str, IntervalTree]
    """
    B_index: List[Tuple[str,int,int]] = []
    chrom_to_tree: Dict[str, IntervalTree] = {}
    for chrom, intervals in setB.items():
        ivs: List[Interval] = []
        for (s, e) in intervals:
            idx = len(B_index)
            B_index.append((chrom, s, e))
            ivs.append(Interval(s, e, idx))
        # Build a tree
        chrom_to_tree[chrom] = IntervalTree(ivs)
    return B_index, chrom_to_tree


def compute_overlaps_setA_vs_setB(
    setA: Dict[str, List[Tuple[int,int]]],
    setB_index: List[Tuple[str,int,int]],
    chrom_to_tree_B: Dict[str, IntervalTree],
    want_per_B: bool = True,
) -> Tuple[int, Optional[np.ndarray]]:
    """Compute weighted overlaps of Set A vs Set B using B's interval trees.
    Returns total overlap and (optionally) per-B vector of overlap counts.
    """
    total = 0
    per_B = np.zeros(len(setB_index), dtype=np.int64) if want_per_B else None

    for chrom, a_list in setA.items():
        if chrom not in chrom_to_tree_B:
            continue
        tree = chrom_to_tree_B[chrom]
        for (as_, ae_) in a_list:
            overlaps = tree.query(as_, ae_)
            if not overlaps:
                continue
            # Each interval independent: sum overlaps per B interval
            for iv in overlaps:
                bp = overlap_len(as_, ae_, iv.start, iv.end)
                if bp:
                    total += bp
                    if per_B is not None and iv.idx is not None:
                        per_B[iv.idx] += bp
    return total, per_B


# -----------------------------
# Permutations
# -----------------------------

def randomize_setA(
    setA: Dict[str, List[Tuple[int,int]]],
    chrom_lengths: Dict[str, int],
    rng: np.random.Generator,
) -> Dict[str, List[Tuple[int,int]]]:
    """Return a new dict with Set A intervals randomly re-positioned on the same chrom,
    preserving length and staying within bounds.
    Overlaps among permuted A intervals are allowed (no restriction in spec).
    """
    perm: Dict[str, List[Tuple[int,int]]] = {}
    for chrom, a_list in setA.items():
        clen = chrom_lengths.get(chrom)
        if clen is None or clen <= 0:
            # If chromosome not in FAI, skip keeping empty
            continue
        out_list: List[Tuple[int,int]] = []
        for (s, e) in a_list:
            length = e - s
            if length >= clen:
                # Clamp: place it at [0, clen)
                new_s = 0
                new_e = clen
            else:
                # valid start range: [0, clen - length]
                max_start = clen - length
                # numpy Generator.integers upper bound exclusive, so +1
                new_s = int(rng.integers(0, max_start + 1))
                new_e = new_s + length
            out_list.append((new_s, new_e))
        perm[chrom] = out_list
    return perm


# -----------------------------
# Main execution
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Genomic overlap permutation test")
    parser.add_argument("setA_bed")
    parser.add_argument("setB_bed")
    parser.add_argument("genome_fai")
    parser.add_argument("output_dir")
    parser.add_argument("num_permutations", nargs="?", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read inputs
    setA = read_bed(args.setA_bed)
    setB = read_bed(args.setB_bed)
    chrom_lengths = read_fai_lengths(args.genome_fai)

    # Basic stats
    setA_regions = sum(len(v) for v in setA.values())
    setB_regions = sum(len(v) for v in setB.values())
    setA_total = total_bases(setA)
    setB_total = total_bases(setB)

    # Build Set B interval trees and index for per-region accounting
    B_index, chrom_to_tree_B = build_B_structures(setB)

    # Observed overlaps
    observed_total, observed_per_B = compute_overlaps_setA_vs_setB(setA, B_index, chrom_to_tree_B, want_per_B=True)

    # Permutation testing
    rng = np.random.default_rng(args.seed)
    num_perm = int(args.num_permutations)

    # Global p-value counter and per-region ge counters
    global_ge = 0
    perB_ge = np.zeros(len(B_index), dtype=np.int64)

    for _ in range(num_perm):
        permA = randomize_setA(setA, chrom_lengths, rng)
        perm_total, perm_per_B = compute_overlaps_setA_vs_setB(permA, B_index, chrom_to_tree_B, want_per_B=True)
        if perm_total >= observed_total:
            global_ge += 1
        # Per-region greater-or-equal counts
        # Compare permuted per-B to observed per-B
        # Note: both are np arrays of int64
        perB_ge += (perm_per_B >= observed_per_B)

    # P-values with +1 pseudocount
    global_p = (global_ge + 1.0) / (num_perm + 1.0)
    perB_p = (perB_ge.astype(np.float64) + 1.0) / (num_perm + 1.0)

    # Bonferroni
    bonf_threshold = 0.05 / float(setB_regions if setB_regions > 0 else 1)
    significant_flags = perB_p < bonf_threshold

    # Write outputs
    # Summary TSV
    summary_rows = [
        ("observed_overlap", int(observed_total)),
        ("global_p_value", float(f"{global_p:.10f}")),
        ("num_permutations", int(num_perm)),
        ("setA_regions", int(setA_regions)),
        ("setB_regions", int(setB_regions)),
        ("setA_total_bases", int(setA_total)),
        ("setB_total_bases", int(setB_total)),
        ("bonferroni_threshold", float(f"{bonf_threshold:.10f}")),
        ("significant_regions_bonferroni", int(significant_flags.sum())),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_path = os.path.join(args.output_dir, "results.tsv")
    df_summary.to_csv(summary_path, sep='\t', index=False)

    # Per-region TSV
    # B_index order is the output order; sort by chrom then start as required
    # Construct a DataFrame then sort
    df_regions = pd.DataFrame(
        {
            "chrom": [c for (c, s, e) in B_index],
            "start": [s for (c, s, e) in B_index],
            "end":   [e for (c, s, e) in B_index],
            "observed_overlap": observed_per_B if observed_per_B is not None else np.zeros(len(B_index), dtype=np.int64),
            "p_value": perB_p,
            "significant_bonferroni": significant_flags,
        }
    )
    df_regions.sort_values(by=["chrom", "start", "end"], inplace=True, kind="mergesort")
    per_region_path = os.path.join(args.output_dir, "results_per_region.tsv")
    df_regions.to_csv(per_region_path, sep='\t', index=False)

    # Console summary
    print(f"Observed weighted overlap: {observed_total}")
    print(f"Global p-value: {global_p:.6g} (permutations={num_perm})")
    print(f"Bonferroni threshold: {bonf_threshold:.6g}; significant regions: {int(significant_flags.sum())}")
    print(f"Wrote: {summary_path}\nWrote: {per_region_path}")


if __name__ == "__main__":
    main()