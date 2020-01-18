import logging
import torch
from prir import pytorch
from prir.path.utils import get_project_root


def get_device(use_gpu=False):
    dev = "cpu"
    if (use_gpu):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            logging.warning("WARING: GPU not available. Fallback to CPU")
    return torch.device(dev)


def run_all_tests(model):
    model.run_train_cross_validation()
    model.reset().run_train()
    model.set_device(get_device(use_gpu=False))
    model.reset().run_train_cross_validation()
    model.reset().run_train()


def main():
    run_all_tests(pytorch.LetterRecognitionModel(device=get_device(use_gpu=True)))
    run_all_tests(pytorch.Cifar10Model(device=get_device(use_gpu=True)))
    run_all_tests(pytorch.Cifar100Model(device=get_device(use_gpu=True)))
    run_all_tests(pytorch.MnistModel(device=get_device(use_gpu=True)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename=get_project_root().joinpath("logfile"),
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    main()
