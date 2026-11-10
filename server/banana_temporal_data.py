"""Training-only episode-tail masks without modifying pinned Observation/loader code."""
import dataclasses
import numpy as np
from flax import struct
from openpi.models import model as model_module


@struct.dataclass
class TemporalObservation(model_module.Observation):
    action_valid_mask: object = None


class EpisodeMaskedDataset:
    """Wrap the already transformed, contiguous LeRobot training dataset."""
    def __init__(self, dataset, episode_lengths, horizon):
        self.dataset = dataset
        self.ends = np.cumsum(episode_lengths, dtype=np.int64)
        self.horizon = horizon
        if any(n < 1 for n in episode_lengths) or len(dataset) != int(self.ends[-1]):
            raise ValueError('Episode mapping/frame count mismatch')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        end = self.ends[np.searchsorted(self.ends, index, side='right')]
        result = dict(self.dataset[index])
        result['action_valid_mask'] = np.arange(self.horizon) < end - index
        return result


class TemporalDataLoader:
    def __init__(self, data_config, data_loader):
        self._data_config, self._data_loader = data_config, data_loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            base = model_module.Observation.from_dict(batch)
            yield TemporalObservation(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                                      action_valid_mask=batch['action_valid_mask']), batch['actions']
