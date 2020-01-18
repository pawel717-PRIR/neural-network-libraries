import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import FitModel as fit
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


crossValidationFlag = True

def main():
    fit.FitCifar100Model(crossValidationFlag)


if __name__ == '__main__':
    main()