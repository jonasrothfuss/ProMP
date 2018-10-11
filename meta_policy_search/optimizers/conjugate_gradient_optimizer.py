from meta_policy_search.utils import logger
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from meta_policy_search.optimizers.base import Optimizer


class FiniteDifferenceHvp(Optimizer):
    def __init__(self, base_eps=1e-5, symmetric=True, grad_clip=None):
        self.base_eps = np.cast['float32'](base_eps)
        self.symmetric = symmetric
        self.grad_clip = grad_clip
        self._target = None
        self.reg_coeff = None
        self._constraint_gradient = None
        self._input_ph_dict = None

    def build_graph(self, constraint_obj, target, input_val_dict, reg_coeff):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            constraint_obj (tf_op) : constraint objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (list) : tuple of tf.placeholders for input data which may be subsampled. The first dimension corresponds to the number of data points
            reg_coeff (float): regularization coefficient
        """
        self._target = target
        self.reg_coeff = reg_coeff
        self._input_ph_dict = input_val_dict

        params = list(target.get_params().values())
        constraint_grads = tf.gradients(constraint_obj, xs=params)

        for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
            if grad is None:
                constraint_grads[idx] = tf.zeros_like(param)

        constraint_gradient = tf.concat([tf.reshape(grad, [-1]) for grad in constraint_grads], axis=0)

        self._constraint_gradient = constraint_gradient

    def constraint_gradient(self, input_val_dict):
        """
        Computes the gradient of the constraint objective

        Args:
            inputs (list): inputs needed to compute the gradient

        Returns:
            (np.ndarray): flattened gradient
        """

        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        constraint_gradient = sess.run(self._constraint_gradient, feed_dict)
        return constraint_gradient

    def Hx(self, input_val_dict, x):
        """
        Compute the second derivative of the constraint val in the direction of the vector x
        Args:
            inputs (list): inputs needed to compute the gradient of the constraint objective
            x (np.ndarray): vector indicating the direction on which the Hessian has to be computed

        Returns: (np.ndarray): second derivative in the direction of x

        """
        assert isinstance(x, np.ndarray)

        param_vals = self._target.get_param_values().copy()
        flat_param_vals = _flatten_params(param_vals)
        eps = self.base_eps
        params_plus_eps_vals = _unflatten_params(flat_param_vals + eps * x, params_example=param_vals)
        self._target.set_params(params_plus_eps_vals)
        constraint_grad_plus_eps = self.constraint_gradient(input_val_dict)
        self._target.set_params(param_vals)

        if self.symmetric:
            params_minus_eps_vals = _unflatten_params(flat_param_vals - eps * x, params_example=param_vals)
            self._target.set_params(params_minus_eps_vals)
            constraint_grad_minus_eps = self.constraint_gradient(input_val_dict)
            self._target.set_params(param_vals)
            hx = (constraint_grad_plus_eps - constraint_grad_minus_eps)/(2 * eps)

        else:
            constraint_grad = self.constraint_gradient(input_val_dict)
            hx = (constraint_grad_plus_eps - constraint_grad)/eps
        return hx

    def build_eval(self, inputs):
        """
        Build the Hessian evaluation function. It let's you evaluate the hessian of the constraint objective
        in any direction.
        Args:
            inputs (list): inputs needed to compute the gradient of the constraint objective

        Returns:
            (function): function that evaluates the Hessian of the constraint objective in the input direction
        """
        def evaluate_hessian(x):
            return self.Hx(inputs, x) + self.reg_coeff * x

        return evaluate_hessian


class ConjugateGradientOptimizer(Optimizer):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.

    Args:
        cg_iters (int) : The number of conjugate gradients iterations used to calculate A^-1 g
        reg_coeff (float) : A small value so that A -> A + reg*I
        subsample_factor (float) : Subsampling factor to reduce samples when using "conjugate gradient. Since the computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        backtrack_ratio (float) : ratio for decreasing the step size for the line search
        max_backtracks (int) : maximum number of backtracking iterations for the line search
        debug_nan (bool) : if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when nan is detected
        accept_violation (bool) : whether to accept the descent step if it violates the line search condition after exhausting all backtracking budgets
        hvp_approach (obj) : Hessian vector product approach
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=0,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            accept_violation=False,
            hvp_approach=FiniteDifferenceHvp(),
            ):

        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks

        self._target = None
        self._max_constraint_val = None
        self._constraint_name = "kl-div"
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        self._hvp_approach = hvp_approach
        self._loss = None
        self._gradient = None
        self._constraint_objective = None
        self._input_ph_dict = None

    def build_graph(self, loss, target, input_ph_dict, leq_constraint):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (list) : tuple of tf.placeholders for input data which may be subsampled. The first dimension corresponds to the number of data points
            extra_inputs (list) : tuple of tf.placeholders for hyperparameters (e.g. learning rate, if annealed)
            leq_constraint (tuple) : A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        """
        assert isinstance(loss, tf.Tensor)
        assert hasattr(target, 'get_params')
        assert isinstance(input_ph_dict, dict)
        
        constraint_objective, constraint_value = leq_constraint

        self._target = target
        self._constraint_objective = constraint_objective
        self._max_constraint_val = constraint_value
        self._input_ph_dict = input_ph_dict
        self._loss = loss

        # build the graph of the hessian vector product (hvp)
        self._hvp_approach.build_graph(constraint_objective, target, self._input_ph_dict, self._reg_coeff)

        # build the graph of the gradients
        params = list(target.get_params().values())
        grads = tf.gradients(loss, xs=params)
        for idx, (grad, param) in enumerate(zip(grads, params)):
            if grad is None:
                grads[idx] = tf.zeros_like(param)
        gradient = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)

        self._gradient = gradient

    def loss(self, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            inputs (list): inputs needed to compute the loss function
            extra_inputs (list): additional inputs needed to compute the loss function

        Returns:
            (float): value of the loss
        """

        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def constraint_val(self, input_val_dict):
        """
        Computes the value of the KL-divergence between pre-update policies for given inputs

        Args:
            inputs (list): inputs needed to compute the inner KL
            extra_inputs (list): additional inputs needed to compute the inner KL

        Returns:
            (float): value of the loss
        """

        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        constrain_val = sess.run(self._constraint_objective, feed_dict)
        return constrain_val

    def gradient(self, input_val_dict):
        """
        Computes the gradient of the loss function

        Args:
            inputs (list): inputs needed to compute the gradient
            extra_inputs (list): additional inputs needed to compute the loss function

        Returns:
            (np.ndarray): flattened gradient
        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        gradient = sess.run(self._gradient, feed_dict)
        return gradient

    def optimize(self, input_val_dict):
        """
        Carries out the optimization step

        Args:
            inputs (list): inputs for the optimization
            extra_inputs (list): extra inputs for the optimization
            subsample_grouped_inputs (None or list): subsample data from each element of the list

        """
        logger.log("Start CG optimization")

        logger.log("computing loss before")
        loss_before = self.loss(input_val_dict)

        logger.log("performing update")

        logger.log("computing gradient")
        gradient = self.gradient(input_val_dict)
        logger.log("gradient computed")

        logger.log("computing descent direction")
        Hx = self._hvp_approach.build_eval(input_val_dict)
        descent_direction = conjugate_gradients(Hx, gradient, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(2.0 * self._max_constraint_val *
                                    (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8)))
        if np.isnan(initial_step_size):
            logger.log("Initial step size is NaN! Rejecting the step!")
            return

        initial_descent_step = initial_step_size * descent_direction
        logger.log("descent direction computed")

        prev_params = self._target.get_param_values()
        prev_params_values = _flatten_params(prev_params)

        loss, constraint_val, n_iter, violated = 0, 0, 0, False
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * initial_descent_step
            cur_params_values = prev_params_values - cur_step
            cur_params = _unflatten_params(cur_params_values, params_example=prev_params)
            self._target.set_params(cur_params)

            loss, constraint_val = self.loss(input_val_dict), self.constraint_val(input_val_dict)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break

        """ ------------------- Logging Stuff -------------------------- """
        if np.isnan(loss):
            violated = True
            logger.log("Line search violated because loss is NaN")
        if np.isnan(constraint_val):
            violated = True
            logger.log("Line search violated because constraint %s is NaN" % self._constraint_name)
        if loss >= loss_before:
            violated = True
            logger.log("Line search violated because loss not improving")
        if constraint_val >= self._max_constraint_val:
            violated = True
            logger.log("Line search violated because constraint %s is violated" % self._constraint_name)

        if violated and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            self._target.set_params(prev_params)

        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")


def _unflatten_params(flat_params, params_example):
    unflat_params = []
    idx = 0
    for key, param in params_example.items():
        size_param = np.prod(param.shape)
        reshaped_param = np.reshape(flat_params[idx:idx+size_param], newshape=param.shape)
        unflat_params.append((key, reshaped_param))
        idx += size_param
    return OrderedDict(unflat_params)


def _flatten_params(params):
    return np.concatenate([param.reshape(-1) for param in params.values()])


def conjugate_gradients(f_Ax, b, cg_iters=10, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b, dtype=np.float32)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))

    return x
