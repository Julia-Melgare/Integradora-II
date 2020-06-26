# --------------------------------------------------------
# Faster R-CNN + Places 365
# Copyright (c) 2020 PUCRS
# Written by Julia Kubiak Melgare
# --------------------------------------------------------

import caffe
import random

class WeightLayer(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 2,            'requires two layer.bottom'
        assert len(top) == 2,               'requires two layer.top'

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        # Copy all of the data
        top[0].data[...] = bottom[0].data[...]
        top[1].data[...] = bottom[1].data[...]
        #print("pre weights:")
        #print(top[0].data[...])
        #print(top[1].data[...])
        #0 = faster | 1 = places
        weights = [0.55, 0.45]
        top[0].data[...] = top[0].data[...]*weights[0]
        top[1].data[...] = top[1].data[...]*weights[1]
        #print("post weights:")
        #print(top[0].data[...])
        #print(top[1].data[...])      

    def backward(self, top, propagate_down, bottom):
        pass