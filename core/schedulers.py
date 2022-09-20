import numpy as np
import torchpack.distributed as dist

__all__ = ['cosine_schedule_with_warmup']


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
    """

    :param k:
    :param num_epochs: configs.num_epochs, max epochs num
    :param batch_size: configs.batch_size
    :param dataset_size: configs.data.training_size
    :return:
    """
    batch_size *= dist.size()

    if dist.size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // dist.size()

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        ratio = (k - warmup_iters) / (num_epochs * iter_per_epoch)
        out = 0.5 * (1 + np.cos(np.pi * ratio))
        return out

