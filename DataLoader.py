import numpy as np
import time
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MyDataset(Dataset):
    def __init__(self, input_ids, input_mask, segment_ids,
                 start_positions=None, end_positions=None, switches=None, answer_mask=None,
                 is_training=False):

        self.input_ids, self.input_mask, self.segment_ids = [torch.cat([i.squeeze(0) \
                for i in input], 0) for input in [input_ids, input_mask, segment_ids]]
        self.is_training = is_training

        if is_training:
            self.start_positions, self.end_positions, self.switches, self.answer_mask = [torch.cat([i.squeeze(0) \
                for i in input], 0) for input in [start_positions, end_positions, switches, answer_mask]]
            indices1, indices2 = [], []
            for i in range(self.input_ids.size(0)):
                switch = [s for (s, m) in zip(self.switches[i], self.answer_mask[i]) if m==1]
                if 3 in switch:
                    indices2.append(i)
                else:
                    indices1.append(i)
            indices = np.random.permutation(range(len(indices2)))
            indices2 = [indices2[i] for i in indices]
            self.positive_indices = indices1
            self.negative_indices = indices2
            self.negative_indices_offset = 0
            self.length = 2*len(self.positive_indices)
        else:
            self.example_index = torch.arange(self.input_ids.size(0), dtype=torch.long)
            self.length = self.input_ids.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_training:
            if idx%2==0:
                idx=self.positive_indices[int(idx/2)]
            else:
                if self.negative_indices_offset==len(self.positive_indices):
                    indices = np.random.permutation(range(len(self.negative_indices)))
                    self.negative_indices = [self.negative_indices[i] for i in indices]
                    self.negative_indices_offset = 0
                else:
                    self.negative_indices_offset+=1
                idx = self.negative_indices[int(idx/2)]
            return [b[idx] for b in [self.input_ids, self.input_mask, self.segment_ids,
                                     self.start_positions, self.end_positions, self.switches, self.answer_mask]]
        return [b[idx] for b in [self.input_ids, self.input_mask, self.segment_ids,
                                     self.example_index]]


class MyDataLoader(DataLoader):

    def __init__(self, features, batch_size, is_training):
        all_input_ids = [torch.tensor([f.input_ids for f in _features], dtype=torch.long) \
                                for _features in features]
        all_input_mask = [torch.tensor([f.input_mask for f in _features], dtype=torch.long) \
                                for _features in features]
        all_segment_ids = [torch.tensor([f.segment_ids for f in _features], dtype=torch.long) \
                                for _features in features]

        if is_training:
            all_start_positions = [torch.tensor([f.start_position for f in _features], dtype=torch.long) \
                                        for _features in features]
            all_end_positions = [torch.tensor([f.end_position for f in _features], dtype=torch.long) \
                                        for _features in features]
            all_switches = [torch.tensor([f.switch for f in _features], dtype=torch.long) \
                                        for _features in features]
            all_answer_mask = [torch.tensor([f.answer_mask for f in _features], dtype=torch.long) \
                                        for _features in features]
            dataset = MyDataset(all_input_ids, all_input_mask, all_segment_ids,
                    all_start_positions, all_end_positions, all_switches, all_answer_mask,
                    is_training=is_training)
            sampler=RandomSampler(dataset)
        else:
            dataset = MyDataset(all_input_ids, all_input_mask, all_segment_ids,
                                is_training=is_training)
            sampler=SequentialSampler(dataset)

        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)

