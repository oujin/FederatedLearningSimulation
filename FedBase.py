import numpy as np
import tensorflow as tf


class FedBase(object):
    def __init__(self, BATCH_SIZE=20, EPOCH_NUM=1,
                 local_lr=0.01, drop_r=0):
        """
        drop_r：模拟设备算力的不一致，即stragglers的比例
        """
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH_NUM = EPOCH_NUM
        self._reset_list()
        self.local_lr = local_lr
        self.drop_r = drop_r
        # 是指上一轮的增量
        self.updates = []

    def _reset_list(self):
        """重置列表"""
        self.global_weights = []
        self.delta_list = []
        self.n_list = []
        self.loss_list = []
        self.acc_list = []

    def metrics(self, y_true, y_pred):
        """测量性能"""
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
        acc = np.mean(y_true == np.argmax(y_pred, axis=1))
        return loss, acc

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
            optimizer.apply_gradients(zip(
                tf.nest.flatten(grads), tf.nest.flatten(model.weights)))
            t_loss += loss.numpy() * y[s:e].shape[0]
            t_acc += acc * y[s:e].shape[0]
        if cnt <= 0:
            return 0, 0, 0
        return cnt, t_loss / cnt, t_acc / cnt

    def set_global_weights(self, weights):
        """设置global weights"""
        self.global_weights = [w + 0 for w in weights]

    def client_train(self, client, x, y, model, r):
        """train on client"""
        # model 指 local model
        self.set_global_weights(model.weights)
        # 保证不同算法得到同样的随机结果
        np.random.seed((r + self.str2int(client)) % (2 ** 32))
        epoch = self.EPOCH_NUM
        isDropped = np.random.choice(
            [True, False], p=[self.drop_r, 1-self.drop_r])
        if isDropped and self.EPOCH_NUM > 1:
            epoch = np.random.randint(1, self.EPOCH_NUM)
        for _ in range(epoch):
            cnt, loss, acc = self.local_train(model, x, y)
        loss, acc = self.metrics(y, model(self.preprocessing(x)))
        # 求model的增量
        self.delta_list.append([
            loc - g for loc, g in zip(model.weights, self.global_weights)
        ])
        self.n_list.append(cnt * epoch)
        self.loss_list.append(loss.numpy())
        self.acc_list.append(acc)

    def avg(self, g_lr=1.0):
        """平均"""
        loss, acc = 0, 0
        delta_weights = [w * 0 for w in self.global_weights]
        # 合并所有增量
        n_sum = sum(self.n_list)
        for d, n, los, a in zip(self.delta_list, self.n_list,
                                self.loss_list, self.acc_list):
            delta_weights = [
                s + (lw * n / n_sum) for s, lw in zip(delta_weights, d)
            ]
            loss = loss + los * n / n_sum
            acc = acc + a * n / n_sum
        # 更新glocal_model的参数
        weights = [
            g + (d * g_lr) for d, g in zip(
                delta_weights, self.global_weights)
        ]
        # 更新update
        self.updates = delta_weights
        # 重置列表
        self._reset_list()
        # 返回权重
        return weights, float(loss), float(acc)

    def fed_eval(self, model, x, y):
        """测试"""
        cnt, acc, loss = y.shape[0], 0, 0
        for i in range(int(np.ceil(cnt / self.BATCH_SIZE))):
            # 求梯度
            s, e = i * self.BATCH_SIZE, (i + 1) * self.BATCH_SIZE
            pred = model(self.preprocessing(x[s:e]))
            metrics = self.metrics(y[s:e], pred)
            loss += metrics[0] * y[s:e].shape[0]
            acc += metrics[1] * y[s:e].shape[0]
        return float(loss) / cnt, float(acc) / cnt, cnt

    def preprocessing(self, x):
        """预处理，如标准化"""
        data = x.astype(np.float32)
        mu, sigma = np.mean(data, 0), np.std(data, 0)
        return (data - mu) / (sigma + 1e-3)

    def str2int(self, string):
        res = 1
        for c in string:
            res *= ord(c)
        return res
