import tensorflow as tf
import tensorflow_probability as tfp


class QuantileClip:
    def __init__(self, clip_quantile: float = 0.9, history_size: int = 1000):
        self.clip_quantile = clip_quantile * 100
        self.grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self.i = tf.Variable(0, trainable=False)
        self.history_size = history_size

    def __call__(self, grads_and_vars):
        grad_norms = [self._get_grad_norm(g) for g, _ in grads_and_vars]
        total_norm = tf.norm(grad_norms)
        assign_idx = tf.math.mod(self.i, self.history_size)
        self.grad_history = self.grad_history[assign_idx].assign(total_norm)
        self.i = self.i.assign_add(1)
        clip_value = tfp.stats.percentile(
            self.grad_history[: self.i], q=self.clip_quantile
        )
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    def _get_grad_norm(self, t, axes=None, name=None):
        values = tf.convert_to_tensor(
            t.values if isinstance(t, tf.IndexedSlices) else t, name="t"
        )

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))
