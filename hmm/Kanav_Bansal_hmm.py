import sys
import math
from collections import defaultdict

# State indices
INBRED = 0
OUTBRED = 1

ERROR_RATE = 1.0 / 1000.0

# Per-base transition rates
L_I_TO_O = 1.0 / (1.5e6)  # inbred -> outbred
L_O_TO_I = 1.0 / (4e6)    # outbred -> inbred


def log(x: float) -> float:
    """Safe log: log(0) -> -inf."""
    if x <= 0.0:
        return float("-inf")
    return math.log(x)


def parse_info_af(info: str):
    """Try to extract AF from the INFO field."""
    for field in info.split(";"):
        if field.startswith("AF="):
            try:
                return float(field.split("=", 1)[1].split(",")[0])
            except ValueError:
                return None
    return None


def genotype_type(gt: str):
    """
    Convert a GT string into 'hom', 'het', or None (for missing/unsupported).

    Assumes diploid genotypes separated by '/' or '|'.
    """
    if gt is None or gt == ".":
        return None
    if gt.startswith("."):
        return None

    if "/" in gt:
        a, b = gt.split("/", 1)
    elif "|" in gt:
        a, b = gt.split("|", 1)
    else:
        # Haploid or weird – treat as missing for simplicity
        return None

    if a == "." or b == ".":
        return None

    return "hom" if a == b else "het"


def parse_vcf(path):
    """
    Parse a VCF file.

    Returns:
        samples: list[str]
        sites: list of dicts with keys:
               - 'pos': int
               - 'p'  : float (ref allele freq)
               - 'genotypes': dict sample -> 'hom'/'het'/None
    """
    sites = []
    samples = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            if not line or line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                fields = line.split("\t")
                samples = fields[9:]
                continue

            fields = line.split("\t")
            chrom, pos, _id, ref, alt, qual, flt, info, fmt = fields[:9]
            pos = int(pos)

            sample_fields = fields[9:]
            format_keys = fmt.split(":")

            if "GT" not in format_keys:
                raise ValueError("VCF FORMAT field does not contain GT")

            gt_idx = format_keys.index("GT")

            # Try INFO/AF first
            p = parse_info_af(info)

            genotypes_by_sample = {}
            ref_count = 0
            alt_count = 0

            for name, sample_entry in zip(samples, sample_fields):
                parts = sample_entry.split(":")
                gt_raw = parts[gt_idx] if gt_idx < len(parts) else "./."
                gtype = genotype_type(gt_raw)
                genotypes_by_sample[name] = gtype

                # If AF is not provided, estimate p from genotypes
                if p is None and gtype is not None:
                    if "/" in gt_raw:
                        a, b = gt_raw.split("/", 1)
                    elif "|" in gt_raw:
                        a, b = gt_raw.split("|", 1)
                    else:
                        continue

                    for allele in (a, b):
                        if allele == "0":
                            ref_count += 1
                        elif allele == "1":
                            alt_count += 1
                        # ignore other alleles

            if p is None:
                total = ref_count + alt_count
                if total == 0:
                    p = 0.5  # total fallback
                else:
                    p = float(ref_count) / total

            sites.append(
                {
                    "pos": pos,
                    "p": p,
                    "genotypes": genotypes_by_sample,
                }
            )

    return samples, sites


def emission_probs(p, genotype):
    """
    Return emission probabilities (P(obs | inbred), P(obs | outbred)).

    genotype: 'hom', 'het', or None.
    """
    q = 1.0 - p

    if genotype is None:
        # Missing: equal likelihood; doesn't affect state choice.
        return 1.0, 1.0

    if genotype == "hom":
        return 1.0 - ERROR_RATE, 1.0 - 2.0 * p * q
    elif genotype == "het":
        return ERROR_RATE, 2.0 * p * q
    else:
        return 1.0, 1.0


def transition_matrix(distance: int):
    """
    Continuous-time Markov chain over distance bp.

    Returns 2x2 matrix T[i][j] = P(state_j | state_i).
    """
    if distance <= 0:
        return [[1.0, 0.0],
                [0.0, 1.0]]

    p_I_stay = math.exp(-L_I_TO_O * distance)
    p_O_stay = math.exp(-L_O_TO_I * distance)
    p_I_to_O = 1.0 - p_I_stay
    p_O_to_I = 1.0 - p_O_stay

    return [
        [p_I_stay, p_I_to_O],
        [p_O_to_I, p_O_stay],
    ]


def initial_probs():
    """Stationary distribution of the CTMC as prior."""
    pi_I = L_O_TO_I / (L_I_TO_O + L_O_TO_I)
    pi_O = L_I_TO_O / (L_I_TO_O + L_O_TO_I)
    return [pi_I, pi_O]


def viterbi_for_sample(positions, ps, genotypes):
    """
    Run Viterbi decoding for one sample.

    positions: list[int]
    ps       : list[float] (allele freqs)
    genotypes: list['hom'/'het'/None]

    Returns: list[int] of state indices (INBRED or OUTBRED).
    """
    n = len(positions)
    if n == 0:
        return []

    # Emission log probabilities
    emit_log = []
    for p, g in zip(ps, genotypes):
        e_I, e_O = emission_probs(p, g)
        emit_log.append([log(e_I), log(e_O)])

    pi = initial_probs()
    dp = [[float("-inf")] * 2 for _ in range(n)]
    prev = [[None] * 2 for _ in range(n)]

    # Initialization
    for s in (INBRED, OUTBRED):
        dp[0][s] = log(pi[s]) + emit_log[0][s]

    # Recursion
    for i in range(1, n):
        d = positions[i] - positions[i - 1]
        T = transition_matrix(d)
        logT = [[log(T[a][b]) for b in (INBRED, OUTBRED)]
                for a in (INBRED, OUTBRED)]

        for s in (INBRED, OUTBRED):
            best_val = float("-inf")
            best_state = None
            for prev_state in (INBRED, OUTBRED):
                val = dp[i - 1][prev_state] + logT[prev_state][s]
                if val > best_val:
                    best_val = val
                    best_state = prev_state
            dp[i][s] = best_val + emit_log[i][s]
            prev[i][s] = best_state

    # Termination
    last_state = INBRED if dp[-1][INBRED] > dp[-1][OUTBRED] else OUTBRED
    path = [last_state]

    # Backtrack
    for i in range(n - 1, 0, -1):
        last_state = prev[i][last_state]
        path.append(last_state)

    path.reverse()
    return path


def find_inbred_segments(positions, states):
    """
    From positions and decoded states, return list of
    (start_pos, stop_pos) for contiguous INBRED regions.
    """
    segments = []
    current_start = None
    prev_pos = None

    for pos, state in zip(positions, states):
        if state == INBRED:
            if current_start is None:
                current_start = pos
        else:
            if current_start is not None and prev_pos is not None:
                segments.append((current_start, prev_pos))
                current_start = None
        prev_pos = pos

    # Close segment if we ended in INBRED
    if current_start is not None:
        segments.append((current_start, positions[-1]))

    return segments


def main(path):
    samples, sites = parse_vcf(path)

    positions = [site["pos"] for site in sites]
    ps = [site["p"] for site in sites]

    all_segments = []

    for sample in samples:
        genotypes = [site["genotypes"][sample] for site in sites]
        states = viterbi_for_sample(positions, ps, genotypes)
        segments = find_inbred_segments(positions, states)

        for start, end in segments:
            all_segments.append((sample, start, end))

    # Sort by sample name, then start position
    all_segments.sort(key=lambda x: (x[0], x[1]))

    # Output
    out = sys.stdout
    print("individual\tstart_position\tstop_position", file=out)
    for sample, start, end in all_segments:
        print(f"{sample}\t{start}\t{end}", file=out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: python {sys.argv[0]} input.vcf\n")
        sys.exit(1)
    main(sys.argv[1])