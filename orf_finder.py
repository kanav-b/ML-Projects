#!/usr/bin/env python3
"""
Self-contained ORF finder that:
1) Generates several small DNA FASTA test files.
2) Scans each file for ORFs on all 6 frames.
3) Prints a concise, human-readable ORF summary to the screen.
4) Also prints the transcribed (RNA) ORF sequences in FASTA format.

Definition here: ORF = start codon (default ATG; optionally CTG/TTG) to the first in-frame stop (TAA/TAG/TGA).
Coordinates are reported 1-based, inclusive, relative to the + strand reference coordinate system.

Run:
  python orf_finder_selftest.py            # generates tests/* and prints results
  python orf_finder_selftest.py --no-alt   # disallow CTG/TTG starts
  python orf_finder_selftest.py --min-aa 20
"""

from dataclasses import dataclass
from typing import List, Tuple, Iterator
import argparse, os, sys

STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CANON = {"ATG"}
START_ALT = {"CTG", "TTG"}  # common bacterial alternatives
COMPLEMENT = str.maketrans("ACGTRYKMSWBDHVNacgtrykmswbdhvn",
                           "TGCAYRMKSWVHDBNtgcayrmkswvhdbn")

AA_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}

@dataclass
class ORF:
    seqid: str
    strand: str       # '+' or '-'
    frame: int        # 0,1,2 within that strand
    start_1b: int     # 1-based inclusive on + strand coords
    end_1b: int       # 1-based inclusive on + strand coords
    nt_len: int
    aa_len: int
    start_codon: str
    stop_codon: str
    gc: float         # GC content in [0,1]
    dna_seq: str      # 5'->3' in transcription direction for that strand
    rna_seq: str
    protein: str      # translated, up to first stop

def parse_fasta(handle) -> Iterator[Tuple[str, str]]:
    header = None
    chunks = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def gc_fraction(s: str) -> float:
    if not s:
        return 0.0
    g = s.count('G') + s.count('g')
    c = s.count('C') + s.count('c')
    return (g + c) / len(s)

def to_rna(dna: str) -> str:
    return dna.replace("T","U").replace("t","u")

def translate(dna: str) -> str:
    aa = []
    for i in range(0, len(dna)-2, 3):
        codon = dna[i:i+3].upper()
        aa_code = AA_TABLE.get(codon, 'X')
        if aa_code == '*':
            break
        aa.append(aa_code)
    return "".join(aa)

def find_orfs(seqid: str, seq_plus: str, allow_alt: bool, min_aa: int) -> List[ORF]:
    L = len(seq_plus)
    start_set = set(START_CANON) | (START_ALT if allow_alt else set())

    def scan_one_strand(seq_ref: str, strand: str):
        out = []
        for frame in range(3):
            i = frame
            start_pos = None
            start_codon = ""
            while i + 3 <= len(seq_ref):
                codon = seq_ref[i:i+3].upper()
                if start_pos is None:
                    if codon in start_set:
                        start_pos = i
                        start_codon = codon
                else:
                    if codon in STOP_CODONS:
                        if strand == '+':
                            s_plus, e_plus = start_pos, i+2
                        else:
                            s_plus = (L - 1) - (i+2)
                            e_plus = (L - 1) - start_pos
                        out.append((s_plus, e_plus, frame, start_codon, codon, strand))
                        start_pos = None
                        start_codon = ""
                i += 3
        return out

    rc = reverse_complement(seq_plus)
    raw = scan_one_strand(seq_plus, '+') + scan_one_strand(rc, '-')

    orfs: List[ORF] = []
    for s_plus, e_plus, frame, start_codon, stop_codon, strand in raw:
        nt_len = e_plus - s_plus + 1
        if nt_len < 3 or (nt_len // 3) < max(1, min_aa):
            continue
        dna_seq = seq_plus[s_plus:e_plus+1] if strand == '+' else reverse_complement(seq_plus[s_plus:e_plus+1])
        orfs.append(ORF(
            seqid=seqid,
            strand=strand,
            frame=frame,
            start_1b=s_plus+1,
            end_1b=e_plus+1,
            nt_len=nt_len,
            aa_len=nt_len//3,
            start_codon=start_codon,
            stop_codon=stop_codon,
            gc=gc_fraction(dna_seq),
            dna_seq=dna_seq,
            rna_seq=to_rna(dna_seq),
            protein=translate(dna_seq)
        ))
    return sorted(orfs, key=lambda o: (o.seqid, o.start_1b, o.end_1b, o.strand, o.frame))

def wrap60(s: str) -> str:
    return "\n".join(s[i:i+60] for i in range(0, len(s), 60))

def summarize(orfs: List[ORF]) -> None:
    if not orfs:
        print("No ORFs found.\n")
        return
    print("SUMMARY")
    print("seqid\tstrand\tframe\tcoords(1-based)\tlen_nt\tlen_aa\tstart\tstop\tGC%\tprotein_start\tprotein_preview")
    for o in orfs:
        prot_preview = (o.protein[:15] + ("…" if len(o.protein) > 15 else "")) or "-"
        print(f"{o.seqid}\t{o.strand}\t{o.frame}\t{o.start_1b}..{o.end_1b}\t{o.nt_len}\t{o.aa_len}\t"
              f"{o.start_codon}\t{o.stop_codon}\t{round(o.gc*100,1):.1f}\t"
              f"{(o.protein[0] if o.protein else '-')}\t{prot_preview}")
    print()
    print("FASTA (transcribed RNA for each ORF)")
    for o in orfs:
        header = (f">{o.seqid}|strand:{o.strand}|frame:{o.frame}|coords:{o.start_1b}..{o.end_1b}"
                  f"|len_nt:{o.nt_len}|len_aa:{o.aa_len}|start:{o.start_codon}|stop:{o.stop_codon}")
        print(header)
        print(wrap60(o.rna_seq))
    print()

def generate_tests(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tests.fasta")
    # 1) Clear + ORF: ATG ... TAG
    seq_plus = "AAAATGAAAGGGCCCTAGTTT"
    # 2) Clear - ORF: embed RC of (ATG CCC TGA)
    seq_minus = "GGG" + reverse_complement("ATGCCCTGA") + "CCC"
    # 3) Overlapping + ORFs: ATG...TGA with another ATG before stop
    seq_overlap = "CCCATGAAACCCATGAAATGATTTTGA"
    # 4) Alternative start (TTG) leading to TAG
    seq_altstart = "TTGAAACCCAAATAG"
    # 5) Noise/no-ORF
    seq_noise = "GACGACGACGACGACGAC"
    with open(path, "w") as f:
        f.write(">plus_orf\n" + seq_plus + "\n")
        f.write(">minus_orf\n" + seq_minus + "\n")
        f.write(">overlap_plus\n" + seq_overlap + "\n")
        f.write(">alt_start\n" + seq_altstart + "\n")
        f.write(">noise\n" + seq_noise + "\n")
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-dir", default="tests", help="Directory to write generated test FASTA files.")
    ap.add_argument("--min-aa", type=int, default=0, help="Minimum peptide length (aa) to report.")
    ap.add_argument("--no-alt", action="store_true", help="Disallow alternative starts (CTG/TTG).")
    args = ap.parse_args()

    tests_path = generate_tests(args.tests_dir)
    print(f"Generated test FASTA: {tests_path}\n")

    allow_alt = not args.no_alt
    with open(tests_path, "r") as fh:
        seq_count = 0
        total_orfs = 0
        for seqid, raw in parse_fasta(fh):
            seq = "".join([c for c in raw.upper() if c.isalpha()])
            print(f"=== {seqid} (len={len(seq)}) ===")
            orfs = find_orfs(seqid, seq, allow_alt=allow_alt, min_aa=args.min_aa)
            summarize(orfs)
            seq_count += 1
            total_orfs += len(orfs)

    print(f"Scanned {seq_count} sequences; reported {total_orfs} ORFs (min_aa={args.min_aa}, allow_alt={allow_alt}).")

if __name__ == "__main__":
    main()
