## Map Matching with Markov Decision Processes \[Proof of concept\]

Here are provided some example scripts for map matching with markov decision processes and reinforcement learning
algorithms.

The provided code is deprecated, has a bad performance and several design flaws. \
For a much improved version, please review `Map Matching 2` repository.

Still for evaluation purposes, you can execute `map_matching.py` and see the results under `exports` and `images`
folders. We also provide an anonymized version of Floating Car Data (FCD)
from [Map and Route GmbH & Co. KG](https://www.mapandroute.de/) for testing purposes. \
See `prepare_points.py` on how we anonymized the data. \
Moreover scripts for evaluating the ground truth data set from
[Microsoft Research: Hidden Markov Map Matching Through Noise and Sparseness](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/)
are provided. The data needs to be downloaded itself, see `data/ground_truth/newson_krumm` folder. Then it can be used
with `map_matching_newson_krumm.py` script. Beware, this is slow.

There are also several benchmarking and evaluation scripts provided. \
Please feel free to explore them yourself.
