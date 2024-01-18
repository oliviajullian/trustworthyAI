from datasets.synthetic_datasets import *
import numpy as np


seeds = [19]

for seed in seeds:
    np.random.seed(seed)
    d = 12
    W = generate_W(d=d, prob=0.5) # 0.2
    c = np.zeros(d)
    s = np.ones([d]) # s = np.round(np.random.uniform(low=0.5, high=2, size=[d]), 1) different variance
    xs, b_, c_ = gen_data_given_model(W, s, c, n_samples=5000, noise_type='lingam', permutate=True)

    print(xs, b_, c_)