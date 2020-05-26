import json
import numpy as np
import tensorflow_federated as tff
import tensorflow as tf
import os


def get_a_nn(init_weights=None, zero_init=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(
            10, activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])
    model.build()
    # 初始化权重
    if init_weights is not None:
        model.set_weights([w + 0 for w in init_weights])
    elif zero_init:
        model.set_weights([w * 0 for w in model.weights])
    return model


if __name__ == "__main__":
    device = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[-1], True)
    # 用tff提供的数据
    emnist_train, _ = tff.simulation.datasets.emnist.load_data(
        cache_dir="cache"
    )
    model = get_a_nn()
    if not os.path.exists('models/'):
        os.makedirs('models/')
    model.save_weights('models/init.h5')
    clients = []
    client_ids = emnist_train.client_ids
    for _ in range(100):
        clients.append(list(
            np.random.choice(client_ids, 10, replace=False)
        ))
    if not os.path.exists('datasets/'):
        os.makedirs('datasets/')
    with open('datasets/clients.json', 'w', encoding='utf8') as f:
        json.dump(clients, f)
