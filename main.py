import torch
import pandas as pd

from utils import (
    get_esm_model,
    load_fasta,
    load_distance_matrix_from_cif,
    MeanEmbedding,
    prepare_sequences_for_esm,
)


def get_distances(dist_matrix: pd.DataFrame, prediction_index: int) -> torch.Tensor:
    """This function should return the distances of the tokens in the sequence to the prediction index
    args:
        dist_matrix: pd.DataFrame, the distance matrix
        prediction_index: int, the index of the prediction
    returns:
        torch.Tensor, the distances of the tokens in the sequence to the prediction index
    """
    return dist_matrix.iloc[prediction_index]


def get_feature_importance(tokens: torch.Tensor, prediction_index: int) -> torch.Tensor:
    """This function should calcuale the feture importance of each token for the prediction of the embedding at a given position
    args:
        tokens: torch.Tensor, the tokens of the sequence
        prediction_index: int, the index of the prediction
    returns:
        torch.Tensor, the feature importance of each token
    """
    # TODO: implement this function
    raise NotImplementedError("This function is not implemented")


if __name__ == "__main__":
    model, alphabet, layers, hidden_dim = get_esm_model(
        "8M"
    )  # 8M is to get things to run faster, for the actual run use 650M.
    seq = load_fasta("brca1")
    dist_matrix = load_distance_matrix_from_cif("brca1")
    dist_index = dist_matrix.index
    seq_dm = "".join([seq[i] for i in dist_index])

    # Use the batch processing function
    toks = prepare_sequences_for_esm([seq_dm], alphabet)
    for i in range(len(seq_dm)):
        # this is how we calculate the mean embedding for for token at index i
        mean_embedding = MeanEmbedding(model, alphabet, i, layers)
        embedding = mean_embedding(toks)
        # distance for all positions for token at index i
        distances = get_distances(dist_matrix, i)
        assert distances[i] == 0
        feature_importance = get_feature_importance(toks, i)
        # TODO: compare the two for correlation (should be negative)
