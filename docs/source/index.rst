.. maml-zoo documentation master file, created by
   sphinx-quickstart on Mon Aug 13 09:57:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Meta-Policy Search's documentation!
====================================

Despite recent progress, deep reinforcement learning (RL) still relies heavily on hand-crafted features and reward functions
as well as engineered problem specific inductive bias. Meta-RL aims to forego such reliance by acquiring inductive bias
in a data-driven manner. A particular instance of meta learning that has proven successful in RL is gradient-based meta-learning.

The code repository provides implementations of various gradient-based Meta-RL methods including

- ProMP: Proximal Meta-Policy Search (`Rothfuss et al., 2018`_)
- MAML: Model Agnostic Meta-Learning (`Finn et al., 2017`_)
- E-MAML: Exploration MAML (`Al-Shedivat et al., 2018`_, `Stadie et al., 2018`_)

The code was written as part of ProMP_. Further information and experimental results can be found on our website_.
This documentation specifies the API and interaction of the algorithm's components. Overall, on iteration of
gradient-based Meta-RL consists of the followings steps:

1. Sample trajectories with pre update policy
2. Perform gradient step for each task to obtain updated/adapted policy
3. Sample trajectories with the updated/adapted policy
4. Perform a meta-policy optimization step, changing the pre-updates policy parameters

This high level structure of the algorithm is implemented in the Meta-Trainer class. The overall structure and interaction
of the code components is depicted in the following figure:


.. image:: MAMLlogic.png
   :width: 600

.. _ProMP: https://arxiv.org/abs/1810.06784

.. _Rothfuss et al., 2018: https://arxiv.org/abs/1810.06784

.. _Finn et al., 2017: https://arxiv.org/abs/1703.03400

.. _Stadie et al., 2018: https://arxiv.org/pdf/1803.01118.pdf

.. _Al-Shedivat et al., 2018: https://arxiv.org/abs/1710.03641

.. _website: https://sites.google.com/view/pro-mp/

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules/meta_policy_search.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
