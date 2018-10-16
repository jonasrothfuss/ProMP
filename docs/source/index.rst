.. maml-zoo documentation master file, created by
   sphinx-quickstart on Mon Aug 13 09:57:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Meta-Policy Search's documentation!
====================================

Meta-Learning, or learning to learn, aims to find a policy that leads to high reward on a randomly sampled task after one (or more) policy updates. Here we implement a variety of policy gradient methods to perform steps 2. and 4. below.

The steps are as follows: 

1. Sample trajectories with pre update policy
2. Perform gradient step for each task to obtain updated policy
3. Sample trajectories with the updated policy
4. Perform gradient step on pre update policy


This code was written as part of ProMP_. Further information can be found on the website_.

.. image:: MAMLlogic.png
   :width: 600

.. _ProMP: http://www.python.org/

.. _website: https://sites.google.com/view/pro-mp/

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules/meta_policy_search.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
