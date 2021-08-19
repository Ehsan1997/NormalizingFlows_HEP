# DOCUMENTATION

## Data Loading API

### dataset.CustomTrainLoaderLHC
```python
from dataset import CustomTrainLoaderLHC
dl = CustomTrainLoaderLHC('Datasets/events_anomalydetection_tiny_table.h5', shuffle=False)

for data, label in dl:
    # DO SOMETHING
    pass
```