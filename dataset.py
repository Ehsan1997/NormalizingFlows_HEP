"""
This file has two main components.
1. Function `fixed_to_table`, which converts the fixed form h5 to table form h5.
Usage: fixed_to_table(example.h5)
2. Class CustomTrainLoaderLHC, this class is an optimized trainloader that works on table form h5 files.
USage: dataloader = CustomTrainLoaderLHC('some_table_hdf.h5')
"""
import pandas as pd
import torch


def fixed_to_table(fixed_file, output_file=None, chunksize=10000):
    """
    Convert fixed format hdf5 file to table format for fast random access.
    Args:
        fixed_file -> Name of the input file
        output_file -> Name of the output file (If not provided, will use the input file_name with 'table' appended to its
    name)
        chunksize -> Number of rows to process at a time (Depends on the amount of RAM available). (Default: 10000)
    """
    if output_file is None:
        file_name, file_extension = fixed_file.split('.')
        output_file = file_name + '_table.' + file_extension

    store = pd.HDFStore(fixed_file)

    nrows = store.get_storer('df').shape[0]

    i = 0
    while i < nrows:
        al_df = store.select(key='df', start=i, stop=i+chunksize)
        if i == 0:
            al_df.to_hdf(output_file, 'df', mode='w', format='table')
        else:
            al_df.to_hdf(output_file, 'df', mode='a', append=True, format='table')
        i += chunksize

    store.close()

    return output_file


class LHCAnomalyDataset(torch.utils.data.Dataset):
    """LHC 2020 R and D dataset for Anomaly Detection"""

    def __init__(self, hdf_file):
        """
        Args:
            hdf_file_file (string): Path to the h5 file with binary label.
        """
        self.hdf_store = pd.HDFStore(hdf_file, mode='r')

    def __len__(self):
        return self.hdf_store.get_storer('df').nrows

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
          idx = [idx]

        fetched_data = torch.tensor(pd.read_hdf(self.hdf_store, 'df', where=pd.Index(idx)).values)

        return fetched_data[:, :-1], fetched_data[:, -1]

    def __del__(self):
        self.hdf_store.close()


class CustomTrainLoaderLHC:
    def __init__(self, file_name, batch_size=500):
        self.ds = LHCAnomalyDataset(file_name)
        self.sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(self.ds),
            batch_size=batch_size,
            drop_last=False)
        self.sampler_iter = iter(self.sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            indices = next(self.sampler_iter)
        except StopIteration:
            self.sampler_iter = iter(self.sampler)
            raise StopIteration
        return self.ds[indices]


if __name__ == '__main__':
    dl = CustomTrainLoaderLHC('Datasets/events_anomalydetection_tiny_table.h5')
    n_epochs = 2
    for epoch in range(n_epochs):
        print(f'Epochs: {epoch}')
        for i, (data, label) in enumerate(dl):
            print('LOL')
            if i == 0:
                print(data[0])
