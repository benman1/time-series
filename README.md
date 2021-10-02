# deepar

[![Build Status](https://travis-ci.com/arrigonialberto86/deepar.svg?branch=master)](https://travis-ci.com/arrigonialberto86/deepar)

Tensorflow implementation of Amazon DeepAR

## Example usage:
Fit a univariate time series:
```python
%load_ext autoreload
%autoreload 2

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from deepar.dataset.time_series import MockTs
from deepar.model.lstm import DeepAR

ts = MockTs(dimensions=1)  # you can change this for multivariate time-series!
dp_model = DeepAR(ts, epochs=50)
dp_model.instantiate_and_fit()
```

Plot results with uncertainty bands:
```python
%matplotlib inline
from numpy.random import normal
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

batch = ts.next_batch(1, ts.n_steps)

ress = []
for i in tqdm.tqdm(range(300)):
    ress.append(np.expand_dims(
        dp_model.get_sample_prediction(
            batch[0]
        ), axis=0,
    ))

res_np = np.concatenate(ress, axis=0)
fig = plt.figure(figsize=(12, 10))

for dim in range(ts.dimensions):
    ax = fig.add_subplot(ts.dimensions, 1, dim+1)
    res_df = pd.DataFrame(res_np[:, :, 0]).T
    tot_res = res_df

    ax.plot(batch[1].reshape((ts.n_steps, ts.dimensions))[:, dim], linewidth=6)
    tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
    tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
    tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
    tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
    tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)

    ax.plot(tot_res.mu, 'bo')
    ax.plot(tot_res.mu, linewidth=2)
    ax.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
    ax.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
    fig.suptitle('Prediction uncertainty')

```

![Image of gaussian](imgs/prediction.png)
