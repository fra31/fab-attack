# FAB: a Fast Adaptive Boundary Attack

This is the code relative to the method introduced in

**Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack**\
Francesco Croce, Matthias Hein\
*University of Tübingen*\
[https://arxiv.org/pdf/1907.02044.pdf](https://arxiv.org/pdf/1907.02044.pdf)

We propose a new white-box adversarial attack against neural networks-based classifiers. FAB-attack aims at changing the
classification of a clean input applying a perturbation
with minimal Lp-norm, for p in {1, 2, inf}. It achieves quickly good quality results, does not need the specification of a step size
and tries to track the desicion boundary.

## News

+ The paper is accepted at ICML 2020!
+ FAB attack is included in [AutoAttack](https://github.com/fra31/auto-attack), a new parameter-free protocol to evaluate adversarial robustness!
+ An optimized PyTorch implementation of FAB attack is now available [here](https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/fast_adaptive_boundary.py) in [Advertorch](https://github.com/BorealisAI/advertorch)!

## Running the attack

We provide [here](https://drive.google.com/file/d/1VBYsfON-lo_JQpmaRezSYxmnePmr49FN/view?usp=sharing), in the folder `models`, classifiers on MNIST and CIFAR-10, trained with either natural training (*plain*), adversarial training
wrt the L2-norm (*l2-at*) or wrt the Linf-norm (*linf-at*). In the folder `datasets`, available at the same link, we provide also the datasets in the format consistent with the scripts.

With

`python test_attack.py --dataset mnist --model plain --bs 1000 --im 1000 --p linf --n_iter 100 --n_restarts 3 --eps 0.3`

one would run FAB-attack on the *plain* model on MNIST wrt the Linf-norm (that is the attack aims at minimizing the Linf-norm of the
adversarial perturbations), using 100 iterations and 3 restarts (`eps` defines the region where to sample the random starting points).
It returns adversarial examples for the first 1000 images of the test set.
More informations about the parameters are available in `test_attack.py`.

The FAB-attack is implemented in `FAB_linf.py`, `FAB_l2.py` and `FAB_l1.py`.
In order to run the attack on other classifiers, it is sufficient to define a model as in `utils.Model`. Then, e.g.,
`FABattack_linf(model, x_input, y_input, sess, hps)`
performs the Linf attack on the model.

## Results

FAB-attack achieves top results in the two challenges at
[https://github.com/yaodongyu/TRADES](https://github.com/yaodongyu/TRADES) and the one at [https://github.com/MadryLab/cifar10_challenge](https://github.com/MadryLab/cifar10_challenge).

## Citations
```
@inproceedings{croce2020minimally,
  author    = {F. Croce and M. Hein},
  title     = {Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack},
  booktitle = {ICML},
  year      = {2020}
}
```
