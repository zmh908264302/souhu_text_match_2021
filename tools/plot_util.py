import random

import matplotlib.pyplot as plt

# 无图形界面需要加，否则plt报错
plt.switch_backend('agg')


def loss_acc_plot(evaluations, path=None):
    train_loss = evaluations[0]
    train_acc = evaluations[1]
    train_f1 = evaluations[2]

    eval_loss = evaluations[3]
    eval_acc = evaluations[4]
    eval_f1 = evaluations[5]

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 2, 1)
    plt.title('loss during train and evaluation')
    plt.xlabel('step')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(2, 2, 2)
    plt.title('accuracy during train and evaluation')
    plt.xlabel('step')
    plt.ylabel('acc')
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, eval_acc)
    plt.legend(['train_acc', 'eval_acc'])

    fig.add_subplot(2, 2, 3)
    plt.title('f1 during train and evaluation')
    plt.xlabel('step')
    plt.ylabel('f1')
    epochs = range(1, len(eval_f1) + 1)
    plt.plot(epochs, train_f1)
    plt.plot(epochs, eval_f1)
    plt.legend(['train_f1', 'eval_f1'])

    plt.savefig(path)


def loss_acc_f1_plot(history: dict, path=None):
    train_loss = history["train_loss"]
    eval_loss = history["eval_loss"]

    train_acc_obj = history["train_acc_obj_class_word"]
    train_acc_express = history["train_acc_express_word"]
    eval_acc_obj = history["eval_acc_obj_class_word"]
    eval_acc_express = history["eval_acc_express_word"]

    train_f1_obj = history["train_f1_obj_class_word"]
    train_f1_express = history["train_f1_express_word"]
    eval_f1_obj = history["eval_f1_obj_class_word"]
    eval_f1_express = history["eval_f1_express_word"]

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 2, 1)
    plt.title('loss during train and evaluation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(2, 2, 2)
    plt.title('acc during train and evaluation')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    epochs = range(1, len(train_acc_obj) + 1)
    plt.plot(epochs, train_acc_obj)
    plt.plot(epochs, train_acc_express)
    plt.plot(epochs, eval_acc_obj)
    plt.plot(epochs, eval_acc_express)
    plt.legend(['train_acc_obj', 'train_acc_express', "eval_acc_obj", "eval_acc_express"])

    fig.add_subplot(2, 2, 3)
    plt.title('f1 during train and evaluation')
    plt.xlabel('epoch')
    plt.ylabel('f1')
    epochs = range(1, len(train_f1_obj) + 1)
    plt.plot(epochs, train_f1_obj)
    plt.plot(epochs, train_f1_express)
    plt.plot(epochs, eval_f1_obj)
    plt.plot(epochs, eval_f1_express)
    plt.legend(['train_f1_obj', 'train_f1_express', "eval_f1_obj", "eval_f1_express"])

    plt.savefig(path)


if __name__ == '__main__':
    history = {
        'train_obj_loss': [random.randint(0, 100) for _ in range(10)],
        'eval_obj_loss': [random.randint(0, 100) for _ in range(10)],
        'eval_obj_acc': [random.randint(0, 100) for _ in range(10)],
        'eval_obj_f1': [random.randint(0, 100) for _ in range(10)]
    }
    loss_acc_plot(history)
