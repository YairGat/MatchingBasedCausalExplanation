import numpy as np
import torch
import torch.nn.functional as F

from utils.constants import DEVICE


def calculate_icace_error(pairs):
    """
    This metric measures the effect of a certain concept on the given model.
    """
    predictions_counterfactual = list(pairs['prediction_counterfactual'])
    predictions_base = list(pairs['prediction_factual'])
    prediction_matching = list(pairs['prediction_match'])

    def icace(prediction_1, prediction_2):
        assert len(prediction_1) == len(prediction_2)
        total_diff = []
        for i in range(len(prediction_1)):
            diff = []
            for j in range(len(prediction_1[0])):
                diff.append(prediction_1[i][j] - prediction_2[i][j])
            total_diff.append(diff)
        return total_diff

    pairs.loc[:, 'icace'] = icace(predictions_counterfactual, predictions_base)

    pairs.loc[:, 'explanation'] = icace(prediction_matching, predictions_base)

    pairs = pairs.reset_index()
    pairs.loc[:, 'icace-error'] = np.array(
        [np.linalg.norm(np.array(pairs['icace'][i]) - np.array(pairs['explanation'][i]), ord=2) for i in
         pairs['icace'].index])

    return pairs


def cosine_similarity_matrix(bases_embeddings, candidates_embeddings):
    dist_mat = np.zeros((len(bases_embeddings), len(candidates_embeddings)))
    for idx_b, base in enumerate(bases_embeddings):
        # base = torch.from_numpy(base).to(DEVICE)
        base = torch.from_numpy(base)
        base = normalize(base)
        for idx_c, candidate in enumerate(candidates_embeddings):
            # candidate = torch.from_numpy(candidate).to(DEVICE)
            candidate = torch.from_numpy(candidate)
            candidate = normalize(candidate)
            dist_mat[idx_b][idx_c] = base @ candidate

    return dist_mat


def cosine_similarity_matrix_2(bases_embeddings, candidates_embeddings):
    if type(bases_embeddings) == torch.Tensor:
        bases = bases_embeddings.to(DEVICE)
    else:
        bases = torch.from_numpy(np.array(bases_embeddings)).to(DEVICE)
    bases = normalize(bases)
    if type(candidates_embeddings) == torch.Tensor:
        candidates = candidates_embeddings.to(DEVICE)
    else:
        candidates = torch.from_numpy(np.array(list(candidates_embeddings.values))).to(DEVICE)
    candidates = normalize(candidates)

    dist_mat = torch.matmul(bases, candidates.T).cpu().numpy()

    return dist_mat


def normalize(x):
    return F.normalize(x, dim=-1)
