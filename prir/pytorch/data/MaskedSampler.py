from torch.utils.data import Sampler


class MaskedSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (i for i in self.mask)

    def __len__(self):
        return len(self.mask)