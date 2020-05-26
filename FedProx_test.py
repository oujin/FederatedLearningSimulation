import datetime
import os
import time
import json

import tensorflow as tf
import tensorflow_federated as tff
from FedProx import FedProx
from sample import get_a_nn

INF = 1000000


def get_data(dataset, client):
    data = dataset.create_tf_dataset_for_client(client)
    data = list(data.batch(INF).as_numpy_iterator())[-1]
    return data['pixels'], data['label']


if __name__ == "__main__":
    device = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[-1], True)

    with open('datasets/clients.json', 'r', encoding='utf8') as f:
        clients = json.load(f)

    fedProx = FedProx(EPOCH_NUM=20, drop_r=0.9)
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        cache_dir="cache"
    )

    test_dataset = emnist_test.create_tf_dataset_from_all_clients()
    test_data = list(test_dataset.batch(INF).as_numpy_iterator())[-1]
    test_x, test_y = test_data['pixels'], test_data['label']

    client_ids = emnist_train.client_ids
    # #
    t_begin = datetime.datetime.now()
    t = time.time()
    c_l_r = 1

    model_prox = get_a_nn()
    model_prox.load_weights('models/init.h5')
    logdir = (f"logs/scalars/fedprox/")
    summary_writer = tf.summary.create_file_writer(logdir)

    r = 0
    for cs in clients:
        # local train
        init_weights = [w + 0 for w in model_prox.weights]
        for client in cs:
            model_prox.set_weights(init_weights)
            train_x, train_y = get_data(emnist_train, client)
            fedProx.client_train(client, train_x, train_y, model_prox, r)
        # global train
        weights, loss, acc = fedProx.avg()
        model_prox.set_weights(weights)
        _time = datetime.datetime.now()
        print(f'[{_time-t_begin}][round:{r}][fedavg]' +
              f'[training] loss={loss},  acc={acc}')
        with summary_writer.as_default():
            tf.summary.scalar("training_loss", loss, step=r)
            tf.summary.scalar("training_acc", acc, step=r)
        # test
        loss, acc, _ = fedProx.fed_eval(model_prox, test_x, test_y)
        _time = datetime.datetime.now()
        print(f'[{_time-t_begin}][round:{r}][fedavg]' +
              f'[testing] loss={loss},  acc={acc}')
        with summary_writer.as_default():
            tf.summary.scalar("testing_loss", loss, step=r)
            tf.summary.scalar("testing_acc", acc, step=r)
        r += 1
