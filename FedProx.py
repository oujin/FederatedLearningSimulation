import numpy as np
import tensorflow as tf
from FedBase import FedBase


class FedProx(FedBase):
    def __init__(self, mu=0.1, **kwarg):
        super(FedProx, self).__init__(**kwarg)
        self.mu = mu

    def local_train(self, model, x, y):
        """local learning
        model: Keras.Model
        x, y: numpy
        """
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]
        optimizer = tf.keras.optimizers.SGD(self.local_lr)
        t_loss, t_acc, cnt = 0, 0, len(indices)
        for i in range(int(np.ceil(cnt / self.BATCH_SIZE))):
            # 求梯度
            s, e = i * self.BATCH_SIZE, (i + 1) * self.BATCH_SIZE
            with tf.GradientTape() as tape:
                pred = model(self.preprocessing(x[s:e]))
                loss, acc = self.metrics(y[s:e], pred)
                grads = tape.gradient(loss, model.weights)
            # 加上扰动约束
            grads = [
                g + self.mu * (w - w0) for g, w, w0 in zip(
                    grads, model.weights, self.global_weights)
            ]
            optimizer.apply_gradients(zip(
                tf.nest.flatten(grads), tf.nest.flatten(model.weights)))
            t_loss += loss.numpy() * y[s:e].shape[0]
            t_acc += acc * y[s:e].shape[0]
        if cnt <= 0:
            return 0, 0, 0
        return cnt, t_loss / cnt, t_acc / cnt
