from maml_zoo.logger import logger
import itertools
import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils

class PerlmutterHvp(object):
    def __init__(self):
        self.target = None
        self.reg_coeff = None

    def update_opt(self, f, target, inputs, reg_coeff):
        assert isinstance(inputs, tuple)
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params()

        constraint_grads = tf.gradients(f, xs=params)
        for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
            if grad is None:
                constraint_grads[idx] = tf.zeros_like(param)

        xs = tuple([tensor_utils.new_tensor_like(p.name.split(":")[0], p) for p in params])

        Hx_plain_splits = tf.gradients(
            tf.reduce_sum(
                tf.stack([tf.reduce_sum(g * x) for g, x in zip(constraint_grads, xs)])
            ),
            params
        )
        for idx, (Hx, param) in enumerate(zip(Hx_plain_splits, params)):
            if Hx is None:
                Hx_plain_splits[idx] = tf.zeros_like(param)
        # Hx_plain_splits = tensor_utils.flatten_tensor_variables(Hx_plain_splits)

        self._all_inputs = inputs + xs
        self.hx_plain = Hx_plain_splits

    def build_eval(self, inputs):
        def eval(x):
            ret = tf.get_default_session().run(self.hx_plain, feed_dict=dict(list(zip(self._all_inputs, inputs + x)))) + self.reg_coeff * x
            return ret

        return eval

# TODO: fix this
class FiniteDifferenceHvp(object):
    def __init__(self, base_eps=1e-8, symmetric=True, grad_clip=None):
        self.base_eps = base_eps
        self.symmetric = symmetric
        self.grad_clip = grad_clip

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff

        params = target.get_params(trainable=True)

        constraint_grads = tf.gradients(f, xs=params)
        for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
            if grad is None:
                constraint_grads[idx] = tf.zeros_like(param)

        flat_grad = tensor_utils.flatten_tensor_variables(constraint_grads)

        def f_Hx_plain(*args):
            inputs_ = args[:len(inputs)]
            xs = args[len(inputs):]
            flat_xs = np.concatenate([np.reshape(x, (-1,)) for x in xs])
            param_val = self.target.get_param_values(trainable=True)
            eps = np.cast['float32'](self.base_eps / (np.linalg.norm(param_val) + 1e-8))
            self.target.set_param_values(param_val + eps * flat_xs, trainable=True)
            flat_grad_dvplus = self.opt_fun["f_grad"](*inputs_)
            self.target.set_param_values(param_val, trainable=True)
            if self.symmetric:
                self.target.set_param_values(param_val - eps * flat_xs, trainable=True)
                flat_grad_dvminus = self.opt_fun["f_grad"](*inputs_)
                hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * eps)
                self.target.set_param_values(param_val, trainable=True)
            else:
                flat_grad = self.opt_fun["f_grad"](*inputs_)
                hx = (flat_grad_dvplus - flat_grad) / eps
            return hx

        self.opt_fun = ext.lazydict(
            f_grad=lambda: tensor_utils.compile_function(
                inputs=inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_Hx_plain=lambda: f_Hx_plain,
        )

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(inputs,xs) + self.reg_coeff * x
            return ret

        return eval


class ConjugateGradientOptimizer(Serializable):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            accept_violation=False,
            hvp_approach=None,
            ):
        """
        Args:
            cg_iters (int) : The number of CG iterations used to calculate A^-1 g
            reg_coeff (float) : A small value so that A -> A + reg*I
            subsample_factor (float) : Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
            backtrack_ratio (float) : 
            max_backtracks (int) : 
            debug_nan (bool) : if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected
            accept_violation (bool) : whether to accept the descent step if it violates the line search condition after
        exhausting all backtracking budgets
            hvp_approach (obj) : 
        """
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks

        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp()
        self._hvp_approach = hvp_approach

    def update_opt(self, loss, target, inputs, extra_inputs=(), leq_constraint, constraint_name="constraint"):
        """
        Sets the objective function and target weights for the optimize function
        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (tuple) : tuple of tf.placeholders for input data which may be subsampled. The first dimension corresponds to the number of data points
            extra_inputs (tuple) : tuple of tf.placeholders for hyperparameters (e.g. learning rate, if annealed)
            leq_constraint (tuple) : A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        """
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)
        
        constraint_term, constraint_value = leq_constraint

        params = target.get_params()
        grads = tf.gradients(loss, xs=params)
        for idx, (grad, param) in enumerate(zip(grads, params)):
            if grad is None:
                grads[idx] = tf.zeros_like(param)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        self._hvp_approach.update_opt(f=constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._all_input_ph = inputs + extra_inputs

        self.loss = loss
        self.flat_grad = flat_grad
        self.constraint_term = constraint_term

    def loss(self, inputs, extra_inputs=()):
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)
        return tf.get_default_session().run(self.loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

    def constraint_val(self, inputs, extra_inputs=()):
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)
        return tf.get_default_session().run(self.constraint_term, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

    def optimize(self, inputs, extra_inputs=(), subsample_grouped_inputs=None):
        prev_param = np.copy(self._target.get_param_values())
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        if self._subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("Start CG optimization: #parameters: %d, #inputs: %d, #subsample_inputs: %d"%(len(prev_param),len(inputs[0]), len(subsample_inputs[0])))
        sess = tf.get_default_session()

        logger.log("computing loss before")
        loss_before = sess.run(self.loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
        logger.log("performing update")

        logger.log("computing gradient")
        flat_g = sess.run(self.flat_grad, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
        logger.log("gradient computed")

        logger.log("computing descent direction")
        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = cg(Hx, flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param)
            loss, constraint_val = sess.run([self.loss, self.constraint_term], feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
            if self._debug_nan and np.isnan(constraint_val):
                import ipdb;
                ipdb.set_trace()
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
            self._max_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" % self._constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                logger.log("Violated because constraint %s is violated" % self._constraint_name)
            self._target.set_param_values(prev_param, trainable=True)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")

def cg(f_Ax, b, cg_iters=10, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
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

    if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x