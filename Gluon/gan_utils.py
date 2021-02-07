
import re, glob 
from matplotlib import pyplot as plt
import mxnet as mx
import numpy as np
import os

class DcGan(object):
    def __init__(self, netG, netD, path = './', prefix = "dcgan"):
        self.netG = netG
        self.netD = netD
        self.path = path
        self.prefix = prefix

    def save_params(self, epoch):
        self.netG.save_params(self._get_params_file(epoch, True))
        self.netD.save_params(self._get_params_file(epoch, False))

    def load_params(self, ctx, epoch = -1):
        if epoch < 0:
            epoch = self._get_last_epoch()

        if epoch > 0: 
            self.netG.load_params(self._get_params_file(epoch, True), ctx=ctx)
            self.netD.load_params(self._get_params_file(epoch, False), ctx=ctx)
        else:
            print("load_params failed ")

        return epoch
    
    def _get_params_file(self, epoch, gen = True):
        if gen:
            file = self.prefix + '-%04d-netG.params' % epoch
        else:
            file = self.prefix + '-%04d-netD.params' % epoch
        return os.path.join(self.path, file) 

    def _get_last_epoch(self):
        epoch = -1 
        ps = glob.glob(self.prefix + "-*-netG.params")
        for p in ps: 
            k = re.match(".*\-(\d+)\-.*params", p)
            v = int(k.groups()[0]) 
            if v > epoch:
                epoch = v
        return epoch 


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

def show_random_images(netG, ctx, num_image = 8, latent_z_size=100):
    for i in range(num_image):
        latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
        img = netG(latent_z)
        plt.subplot(2,4,i+1)
        visualize(img[0])
    plt.show()

def show_progress_images(netG, ctx, num_image = 12, step = 0.05, latent_z_size=100):
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    for i in range(num_image):
        img = netG(latent_z)
        plt.subplot(3,4,i+1)
        visualize(img[0])
        latent_z += step
    plt.show()
