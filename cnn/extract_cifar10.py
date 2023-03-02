# Python 3 program to visualize 4th image
import matplotlib.pyplot as plt
import numpy as np
import cv2



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def upSampleCIFAR10():
    file = r'/fs2/comm/kpgrp/mhosseini/github/ConvSearch/cnn/data/cifar-10-batches-py/data_batch_2'
    data_batch_1 = unpickle(file)


    X_train = data_batch_1['data']
    meta_file = r'/fs2/comm/kpgrp/mhosseini/github/ConvSearch/cnn/data/cifar-10-batches-py/batches.meta'
    meta_data = unpickle(meta_file)
    label_name = meta_data['label_names']

    image = data_batch_1['data'][0]
    image = image.reshape(3,32,32)
    image = image.transpose(1,2,0)
    im = np.uint8(image)
    label = data_batch_1['labels'][0]
    plt.title(label_name[label])
    plt.imshow(im)
    plt.savefig('cifar10.png')
    
    
    
    print(image.shape)

    imgOriginal = image
    imgUpsampled = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    print(imgUpsampled.shape)
    
    return imgOriginal, imgUpsampled