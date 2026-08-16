r"""E. coli genome — a real 4-state sequence for Markov modeling.

The complete K-12 MG1655 genome (~4.6M nucleotides over the alphabet
A, C, G, T) from NCBI: the classic real-world testbed for Markov
chains — genome statistics were Markov-modeled decades before
language was. Downloaded once (~1.4MB gzipped FASTA), parsed, cached.

    from dna import load_ecoli, DNA_ALPHABET
    ids = load_ecoli()          # int64 array, 0..3 = A, C, G, T

Prepared for the next-state-prediction experiment: fit
p(x_{t+1} | last k) by counting and by a classifier, compare held-out
cross-entropy against the entropy-rate story of
Theory/markov-chains/markov-cross-entropy.pdf.
"""

import gzip
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np

_DIR = Path(__file__).resolve().parent / "dna"
_URL = ("https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
        "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz")
DNA_ALPHABET = "ACGT"


def load_ecoli() -> np.ndarray:
    cache = _DIR / "ecoli.npz"
    if not cache.exists():
        _DIR.mkdir(parents=True, exist_ok=True)
        fna = _DIR / "ecoli.fna.gz"
        if not fna.exists():
            print("downloading E. coli K-12 genome (~1.4MB) ...")
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(_URL, context=ctx) as r:
                fna.write_bytes(r.read())
        text = gzip.decompress(fna.read_bytes()).decode("ascii")
        seq = "".join(line for line in text.splitlines()
                      if not line.startswith(">")).upper()
        lut = {c: i for i, c in enumerate(DNA_ALPHABET)}
        ids = np.array([lut[c] for c in seq if c in lut],
                       dtype=np.int64)
        np.savez_compressed(cache, ids=ids)
    return np.load(cache)["ids"]


if __name__ == "__main__":
    ids = load_ecoli()
    print(f"{len(ids):,} nucleotides, "
          f"frequencies {np.bincount(ids) / len(ids)}")
