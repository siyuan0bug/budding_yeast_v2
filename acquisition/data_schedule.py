"""
Data scheduling for Active Learning.
Controls how many batches to generate per AL iteration.
"""

class DataSchedule:
    def __init__(self, first_batch_num):
        self.first_batch_num = first_batch_num

    def num_batches(self, al_iter: int) -> int:
        raise NotImplementedError

    def __call__(self, al_iter):
        return self.num_batches(al_iter)


class LinearSchedule(DataSchedule):
    def num_batches(self, al_iter: int) -> int:
        return self.first_batch_num


class ExponentialSchedule(DataSchedule):
    def num_batches(self, al_iter: int) -> int:
        return self.first_batch_num * (2 ** al_iter)


class FixedSchedule(DataSchedule):
    def __init__(self, batch_num_per_iter):
        self.batch_num_per_iter = list(batch_num_per_iter)
        super().__init__(self.batch_num_per_iter[0])

    def num_batches(self, al_iter: int) -> int:
        return self.batch_num_per_iter[al_iter]
