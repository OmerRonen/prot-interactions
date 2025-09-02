import os
import tempfile
import esm
import torch

import pandas as pd
import numpy as np


from typing import List, Union

from Bio import SeqIO
from Bio.Seq import Seq

from esm.data import FastaBatchedDataset

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data"


class MeanEmbedding(torch.nn.Module):
    def __init__(
        self,
        model: esm.model.esm2.ESM2,
        alphabet: esm.data.Alphabet,
        sequence_index: int,
        layer: int = 33,
    ):
        super(MeanEmbedding, self).__init__()
        self.model = model
        self.alphabet = alphabet
        self.sequence_index = sequence_index
        self.layer = layer

    def forward(self, toks):
        out_esm = self.model(
            toks,
            repr_layers=[self.layer],
            return_contacts=False,
        )["representations"]
        embed = out_esm[self.layer][:, 1:-1, ...]
        embed = embed[:, self.sequence_index, :]
        return embed.mean(axis=-1)


def prepare_sequences_for_esm(
    sequences: List[str], alphabet: esm.data.Alphabet
) -> torch.Tensor:
    """
    Prepare sequences for ESM model using batch processing.

    Args:
        sequences: List of protein sequences
        alphabet: ESM alphabet

    Returns:
        torch.Tensor: Tokenized sequences ready for ESM model
    """
    batch_converter = alphabet.get_batch_converter()
    fasta_tmp = tempfile.NamedTemporaryFile(suffix=".fasta")
    save_fasta_file(sequences, fasta_tmp.name)
    dataset = FastaBatchedDataset.from_file(fasta_tmp.name)
    seq_len = len(sequences[0])
    seqs_per_batch = len(sequences)
    toks_per_batch = seqs_per_batch * seq_len + 1

    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=batch_converter, batch_sampler=batches
    )

    # Get the first (and only) batch directly
    batch_data = next(iter(data_loader))
    (labels, strs, toks) = batch_data
    toks = toks.to(device=DEVICE, non_blocking=True)

    # Clean up temporary file
    fasta_tmp.close()

    return toks


def save_fasta_file(sequences, filename):
    records = [
        SeqIO.SeqRecord(Seq(seq), id=f"Sequence{i + 1}")
        for i, seq in enumerate(sequences)
    ]
    SeqIO.write(records, filename, "fasta")


def get_esm_model(n_params: str) -> tuple:
    """Loads a pre-trained ESM model.

    Args:
        n_params (str): number of parameters or model type identifier.
                        Supported special strings: "1v" for ESM-1v, "2_650M" for ESM-2 650M.

    Returns:
        tuple: (esm_model, alphabet, layers, hidden_dim)
    """

    if n_params == "650M":
        # ESM-2 (650M params)
        layers = 33
        hidden_dim = 1280
        func_name = "esm2_t33_650M_UR50D"
    else:
        # Default behavior from the original code
        if n_params == "15B":
            layers = 48
            hidden_dim = 5120
        elif n_params == "3B":
            layers = 36
            hidden_dim = 2560
        elif n_params == "650M":
            layers = 33
            hidden_dim = 1280
        elif n_params == "150M":
            layers = 30
            hidden_dim = 640
        elif n_params == "35M":
            layers = 12
            hidden_dim = 480
        elif n_params == "8M":
            layers = 6
            hidden_dim = 320
        else:
            raise ValueError(f"Unsupported model parameter setting: {n_params}")

        func_name = f"esm2_t{layers}_{n_params}_UR50D"

        # If you had previously mapped "650M" to esm1b, you can uncomment:
        # if n_params == "650M":
        #     func_name = "esm1b_t33_650M_UR50S"

    # Use the standard ESM model loading
    func = getattr(esm.pretrained, func_name)
    esm_model, alphabet = func()
    esm_model = esm_model.eval().to(DEVICE)
    return esm_model, alphabet, layers, hidden_dim


def load_fasta(
    gene: str, start_idx: Union[int, None] = None, end_idx: Union[int, None] = None
) -> str:
    """
    Load a FASTA file for a given gene and return the protein sequence.

    Args:
        gene: Gene name (used to construct the filename)

    Returns:
        str: The protein sequence from the FASTA file

    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
        ValueError: If the FASTA file is empty or has no sequences
    """
    f_name = os.path.join(DATA_PATH, f"{gene}.fasta")

    if not os.path.exists(f_name):
        raise FileNotFoundError(f"FASTA file not found: {f_name}")

    # Read the FASTA file using Biopython
    try:
        # Read the first (and typically only) sequence from the FASTA file
        with open(f_name, "r") as handle:
            record = next(SeqIO.parse(handle, "fasta"))
            sequence = str(record.seq)

        if not sequence:
            raise ValueError(f"FASTA file is empty: {f_name}")

        if start_idx is not None or end_idx is not None:
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(sequence)
            sequence = sequence[start_idx:end_idx]

        return sequence
    except StopIteration:
        raise ValueError(f"No sequences found in FASTA file: {f_name}")
    except Exception as e:
        raise RuntimeError(f"Error reading FASTA file {f_name}: {str(e)}")


def load_distance_matrix_from_cif(gene: str) -> pd.DataFrame:
    """
    Load a Cα–Cα distance matrix from a CIF file using Biopython and format it with proper row/column names.

    Args:
        f_name: Path to the CIF file

    Returns:
        pd.DataFrame: Distance matrix with row/column names as residue indices (1-based)

    Raises:
        FileNotFoundError: If CIF file doesn't exist
        ValueError: If CIF file format is invalid or residue sequence mismatch
    """
    wt_seq = load_fasta(gene)
    if gene == "brca1":
        f_name = os.path.join(DATA_PATH, "1t15_brca1.cif")
    elif gene == "myh7":
        f_name = os.path.join(DATA_PATH, "2fxm_myh7.cif")
    else:
        raise ValueError(f"Gene {gene} not supported")

    if not os.path.isfile(f_name):
        raise FileNotFoundError(f"CIF file not found: {f_name}")

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", f_name)
    except Exception as e:
        raise ValueError(f"Failed to parse CIF file '{f_name}': {e}")

    # Get the first model and first chain (adjust as needed)
    model = next(structure.get_models())
    chain = next(model.get_chains())

    # Collect Cα atoms and sequence
    ca_atoms = []
    seq = []
    for res in chain:
        if "CA" in res:
            ca_atoms.append(res["CA"])
            seq.append(res.get_resname())

    seq_oneletter = "".join([protein_letters_3to1.get(res, "X") for res in seq])
    # LOGGER.info(f"Sequence from CIF: {seq_oneletter}")
    # find seq in the wild type sequence
    # check if seq_oneletter is in wt_seq
    if seq_oneletter not in wt_seq:
        raise ValueError(
            f"Sequence from CIF: {seq_oneletter} not found in wild type sequence: {wt_seq}"
        )
    start_idx = wt_seq.find(seq_oneletter)
    end_idx = start_idx + len(seq_oneletter)

    # Compute Cα–Cα distances
    n = len(ca_atoms)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = ca_atoms[i] - ca_atoms[j]

    # Make DataFrame with 1-based indices
    residue_indices = np.arange(start_idx, end_idx)
    df = pd.DataFrame(dist_matrix, index=residue_indices, columns=residue_indices)

    return df
