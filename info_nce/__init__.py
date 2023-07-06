import torch
import torch.nn.functional as F
from torch import nn
from utils.constants import DEVICE

# __all__ = ['InfoNCE', 'info_nce']
__all__ = ['InfoNCE']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        # >>> loss = InfoNCE()
        # >>> batch_size, num_negative, embedding_size = 32, 48, 128
        # >>> query = torch.randn(batch_size, embedding_size)
        # >>> positive_key = torch.randn(batch_size, embedding_size)
        # >>> negative_keys = torch.randn(num_negative, embedding_size)
        # >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, coefficients, device, loss_version, temperature=0.1, reduction='mean', negative_mode='unpaired',
                 mean_loss=False):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.coefficients = coefficients
        self.mean_loss = mean_loss
        self.device = device
        self.loss_version = loss_version

    def forward(self, query, positive_keys, negative_keys=None, confounder_counterfactual_keys=None, approx_keys=None,
                eval=False):
        if self.loss_version == 0:
            return self.info_nce_3(query=query, positive_keys=positive_keys,
                                   negative_keys=negative_keys,
                                   confounder_counterfactual_keys=confounder_counterfactual_keys,
                                   approx_keys=approx_keys,
                                   temperature=self.temperature, eval=eval)
        elif self.loss_version == 1:
            return self.info_nce_2(query=query, positive_keys=positive_keys,
                                   negative_keys=negative_keys,
                                   confounder_counterfactual_keys=confounder_counterfactual_keys,
                                   approx_keys=approx_keys,
                                   temperature=self.temperature, eval=eval)

    def info_nce_2(self, query, positive_keys, negative_keys=None, confounder_counterfactual_keys=None,
                   approx_keys=None, temperature=None, eval=False):

        query, positive_key, negative_keys, confounder_counterfactual_keys, approx_keys = normalize(query,
                                                                                                    positive_keys,
                                                                                                    negative_keys,
                                                                                                    confounder_counterfactual_keys,
                                                                                                    approx_keys)
        if eval:
            temperature = 1

        coefficients = {
            't_cf': self.coefficients['t_cf'] if not eval else 1,
            'approx': self.coefficients['approx'] if not eval else 1,
            'negative': self.coefficients['negative'] if not eval else 1,
            'c_cf': self.coefficients['c_cf'] if not eval else 1
        }

        treatment_c_logit = coefficients['t_cf'] * torch.exp(
            torch.matmul(query, transpose(positive_key)) / temperature) if \
            coefficients['t_cf'] != 0 and positive_key is not None else torch.zeros(1, 1).to(DEVICE)

        approx_logit = coefficients['approx'] * torch.exp(
            torch.matmul(query, transpose(approx_keys)) / temperature) if \
            coefficients['approx'] != 0 and approx_keys is not None else torch.zeros(1, 1).to(DEVICE)

        sample_logit = coefficients['negative'] * torch.exp(
            torch.matmul(query, transpose(negative_keys)) / temperature) if \
            coefficients['negative'] != 0 and negative_keys is not None else torch.zeros(1, 1).to(DEVICE)

        confounder_c_logit = coefficients['c_cf'] * torch.exp(
            torch.matmul(query, transpose(confounder_counterfactual_keys)) / temperature) \
            if coefficients['c_cf'] != 0 and confounder_counterfactual_keys is not None else torch.zeros(1, 1).to(
            DEVICE)

        if self.mean_loss:
            positive_logits = torch.mean(treatment_c_logit.float()) + torch.mean(approx_logit.float())
            negative_logits = torch.mean(sample_logit.float()) + torch.mean(confounder_c_logit.float())
        else:
            positive_logits = treatment_c_logit.float().sum() + approx_logit.float().sum()
            negative_logits = sample_logit.float().sum() + confounder_c_logit.float().sum()

        score = -1 * torch.log(positive_logits / (positive_logits + negative_logits))

        return score

    def info_nce_3(self, query, positive_keys, negative_keys=None, confounder_counterfactual_keys=None,
                   approx_keys=None,
                   temperature=0.1, eval=False):

        query, positive_key, negative_keys, confounder_counterfactual_keys, approx_keys = normalize(query,
                                                                                                    positive_keys,
                                                                                                    negative_keys,
                                                                                                    confounder_counterfactual_keys,
                                                                                                    approx_keys)

        if eval:
            temperature = 1

        coefficients = {
            'tcf_cfc': self.coefficients['tcf_cfc'] if not eval else 1,
            'tcf_pax': self.coefficients['tcf_pax'] if not eval else 1,
            'tcf_nax': self.coefficients['tcf_nax'] if not eval else 1,
            'pax_nax': self.coefficients['pax_nax'] if not eval else 1,
            'pax_cfc': self.coefficients['pax_cfc'] if not eval else 1,
            'cfc_nax': self.coefficients['cfc_nax'] if not eval else 1
        }

        tcf_logit = torch.exp(
            torch.matmul(query, transpose(positive_key)) / temperature)

        approx_logit = torch.exp(
            torch.matmul(query, transpose(approx_keys)) / temperature)

        negative_logit = torch.exp(
            torch.matmul(query, transpose(negative_keys)) / temperature)

        cfc_logit = torch.exp(
            torch.matmul(query, transpose(confounder_counterfactual_keys)) / temperature)

        if self.mean_loss:
            tcf_score = torch.mean(tcf_logit.float())
            approx_score = torch.mean(approx_logit.float())
            negative_score = torch.mean(negative_logit.float())
            cfc_score = torch.mean(cfc_logit.float())
        else:
            tcf_score = tcf_logit.float().sum()
            approx_score = approx_logit.float().sum()
            negative_score = negative_logit.float().sum()
            cfc_score = cfc_logit.float().sum()

        score_tcf_cfc = -1 * coefficients['tcf_cfc'] * torch.log(tcf_score / (tcf_score + cfc_score))
        score_tcf_pax = -1 * coefficients['tcf_pax'] * torch.log(
            tcf_score / (tcf_score + approx_score))
        score_tcf_nax = -1 * coefficients['tcf_nax'] * torch.log(
            tcf_score / (tcf_score + negative_score))
        score_pax_cfc = -1 * coefficients['pax_cfc'] * torch.log(
            approx_score / (approx_score + cfc_score))
        score_pax_nax = -1 * coefficients['pax_nax'] * torch.log(
            approx_score / (approx_score + negative_score))
        score_cfc_nax = -1 * coefficients['cfc_nax'] * torch.log(
            cfc_score / (cfc_score + negative_score))

        total_score = (
                score_tcf_cfc + score_tcf_pax + score_tcf_nax + score_pax_cfc + score_pax_nax + score_cfc_nax)

        return total_score

    def info_nce(self, query, positive_keys, negative_keys=None, confounder_counterfactual_keys=None, approx_keys=None,
                 temperature=0.1, eval=False):

        query, positive_key, negative_keys, confounder_counterfactual_keys, approx_keys = normalize(query,
                                                                                                    positive_keys,
                                                                                                    negative_keys,
                                                                                                    confounder_counterfactual_keys,
                                                                                                    approx_keys)
        if not eval:
            c_1 = self.coefficients['t_cf']
            c_2 = self.coefficients['approx']
            c_3 = self.coefficients['negative']
            c_4 = self.coefficients['c_cf']
        else:
            c_1 = 1
            c_2 = 1
            c_3 = 1
            c_4 = 1

        if c_1 == 0 or positive_key is None:
            treatment_c_logit = torch.zeros(1, 1).to(DEVICE)
        else:
            treatment_c_logit = c_1 * torch.exp(
                (query @ transpose(positive_key)) / temperature)

        if c_2 == 0 or approx_keys is None:
            approx_logit = torch.zeros(1, 1).to(DEVICE)
        else:
            approx_logit = c_2 * torch.exp((query @ transpose(approx_keys)) / temperature)

        if c_3 == 0 or negative_keys is None:
            sample_logit = torch.zeros(1, 1).to(DEVICE)
        else:
            sample_logit = c_3 * torch.exp(
                (query @ transpose(negative_keys)) / temperature)

        if c_4 == 0 or confounder_counterfactual_keys is None:
            confounder_c_logit = torch.zeros(1, 1).to(DEVICE)
        else:
            confounder_c_logit = c_4 * torch.exp((query @ transpose(
                confounder_counterfactual_keys)) / temperature)

        if self.mean_loss:
            positive_logits = torch.mean(treatment_c_logit.float()) + torch.mean(approx_logit.float())
            negative_logits = torch.mean(sample_logit.float()) + torch.mean(confounder_c_logit.float())
        else:
            positive_logits = treatment_c_logit.float() + torch.sum(approx_logit.float())
            negative_logits = torch.sum(sample_logit.float()) + torch.sum(confounder_c_logit.float())

        score = -1 * torch.log(positive_logits / (positive_logits + negative_logits))

        return score

    # def info_nce_original(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean',
    #                       negative_mode='unpaired'):
    #     # Check input dimensionality.
    #     # Normalize to unit vectors
    #     query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    #     if negative_keys is not None:
    #         # Explicit negative keys
    #
    #         # Cosine between positive pairs
    #         positive_logit = query @ transpose(positive_key)
    #
    #         negative_logits = query @ transpose(negative_keys)
    #
    #         # First index in last dimension are the positive samples
    #         logits = torch.cat([positive_logit, negative_logits], dim=1)
    #         # labels = torch.cat([torch.ones(len(positive_logit)),
    #         #                     torch.zeros(negative_logits.shape[1])]).long().to(device)
    #         labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    #         # labels = torch.zeros(len(positive_logit), dtype=torch.long, device=query.device)
    #     return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    normalized = []
    for x in xs:
        if x is None or x == "":
            normalized.append(None)
        else:
            normalized.append(F.normalize(x, dim=-1))
    return normalized
