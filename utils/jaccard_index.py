import torch
from torch import Tensor


def jaccard_index(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    print(input.size())
    print(target.size())
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter
    union = torch.where(union == 0, inter, union)

    jaccard = (inter + epsilon) / (union + epsilon)
    return jaccard.mean()


def multiclass_jaccard_index(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return jaccard_index(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def jaccard_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_jaccard_index if multiclass else jaccard_index
    return 1 - fn(input, target, reduce_batch_first=True)
