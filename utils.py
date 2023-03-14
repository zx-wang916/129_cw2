import os


def create_dir():
    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isdir('model'):
        os.mkdir('model')

    if not os.path.isdir('model/supervised'):
        os.mkdir('model/supervised')

