import os
from gpjax import save, load, RBF
from gpjax.gps import Prior
from gpjax.likelihoods import Bernoulli


def test_save_load():
    kern = RBF()
    prior = Prior(kern)

    posterior = prior * Bernoulli()
    save(posterior, 'test')
    assert os.path.exists("./test.npz")
    temp = Prior(kern) * Bernoulli()
    load(temp, './test.npz')
    assert temp == posterior
    os.remove('./test.npz')