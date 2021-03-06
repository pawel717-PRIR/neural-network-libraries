import logging
import math

import torch
from prir.path.utils import get_project_root
from prir.pytorch.model.Cifar100Model import Cifar100Model
from prir.pytorch.model.Cifar10Model import Cifar10Model
from prir.pytorch.model.LetterRecognitionModel import LetterRecognitionModel
from prir.pytorch.model.MnistModel import MnistModel


def get_device(use_gpu=False):
    dev = "cpu"
    if use_gpu:
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            logging.warning("WARING: GPU not available. Fallback to CPU")
    return torch.device(dev)


def run_all_tests(model):
    model.set_device(get_device(use_gpu=True))
    model.run_train_cross_validation()
    model.reset().run_train()
    model.set_device(get_device(use_gpu=False))
    model.reset().run_train_cross_validation()
    model.reset().run_train()


def main():
    run_all_tests(MnistModel())
    run_all_tests(Cifar10Model())
    run_all_tests(Cifar100Model())
    run_all_tests(LetterRecognitionModel())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename=get_project_root().joinpath("logfile"),
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    main()
