"""model of vgg-net 16"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import vgg_base

def get_layer_parents(adjList,lidx):
  """ Returns parent layer indices for a given layer index. """
  return [e[0] for e in adjList if e[1]==lidx]

class VGG16(vgg_base.ConvNet):
    """docstring for VGG16""" 
    def __init__(self, nnObj,is_training,batch_norm_decay,batch_norm_epsilon,data_format='channels_first'):
        super(VGG16,self).__init__(is_training,data_format,batch_norm_decay,batch_norm_epsilon)
        self.nn = nnObj
        self.num_classes = 10+1 
        self.block_num=2
        self.block_filter_sizes = [256,512]


    def _get_layers(self,nn,lidx,num_incoming_filters=None):
        # Inputs:
        #   nn - neural_network object
        #   lidx - layer index
        #   num_incoming_filters - number of filters from parents (after concatenation)
        layerStr = nn.layer_labels[lidx]
        strideVal = nn.strides[lidx]
        poolSizeVal = 3
        stridePoolVal = 2 if lidx!=3 else 1
        num_filters = nn.num_units_in_each_layer[lidx]
        if num_filters==None: num_filters=1
        if layerStr=='relu':
            return lambda x: self._relu_layer(x)
        elif layerStr=='conv3':
            return lambda x: self._conv_layer(x,3,num_filters,strideVal)
        elif layerStr=='conv5':
            return lambda x: self._conv_layer(x,5,num_filters,strideVal)
        elif layerStr=='conv7':
            return lambda x: self._conv_layer(x,7,num_filters,strideVal)
        elif layerStr=='conv9':
            return lambda x: self._conv_layer(x,9,num_filters,strideVal)
        elif layerStr=='res3':
            return lambda x: self._residual_layer(x,3,num_incoming_filters,num_filters,strideVal)
        elif layerStr=='res5':
            return lambda x: self._residual_layer(x,5,num_incoming_filters,num_filters,strideVal)
        elif layerStr=='res7':
            return lambda x: self._residual_layer(x,7,num_incoming_filters,num_filters,strideVal)
        elif layerStr=='res9':
            return lambda x: self._residual_layer(x,9,num_incoming_filters,num_filters,strideVal)
        elif layerStr=='avg-pool':
            return lambda x: self._avg_pool_layer(x,poolSizeVal,stridePoolVal)
        elif layerStr=='max-pool':
            return lambda x: self._max_pool_layer(x,poolSizeVal,stridePoolVal)
        elif layerStr=='fc':
            return lambda x: self._full_connected_layer(x,num_filters)
        elif layerStr=='softmax':
            num_filters=self.num_classes
            return lambda x: self._full_connected_layer(x,num_filters)

    def forward_pass(self,x,input_data_format='channels_last'):
        if self._data_format != input_data_format:
            if input_data_format == 'channels_last':
                x = tf.transpose(x,[0,3,1,2])
            else:
                x = tf.transpose(x,[0,2,3,1])
        x = x/128.0 -1 
        nn = self.nn
        # x = self._conv_layer(x,3,64,1)
        # x = self._conv_layer(x,3,64,1)
        # x = self._max_pool_layer(x,3,1)
        # x = self._conv_layer(x,3,128,1)
        # x = self._conv_layer(x,3,128,1)
        # x = self._max_pool_layer(x,3,2)
        # for bfs in self.block_filter_sizes:
        #     for i in range(self.block_num):
        #         x = self._conv_layer(x,3,bfs,1)
        #     x = self._max_pool_layer(x,3,2)
        # #x = self._full_connected_layer(x,512)
        # #x = self._full_connected_layer(x,256)
        # x = self._full_connected_layer(x,self.num_classes)
        # #x = self._softmax_layer(x)

        # return [x]
        # # Printing next architecture before translating into tensorflow
        print('=================================================')
        print('List of layers and num-units in next architecture:')
        for lidx in range(1,nn.num_internal_layers+1):
            layerToPrint = nn.layer_labels[lidx]
            unitsToPrint = nn.num_units_in_each_layer[lidx]
            print('layer-label = %s,  num-units = %s' % (layerToPrint, unitsToPrint))
        print('=================================================')

        # Loop over layers and define conv net 
        layers = [x]  # Add first layer to layers-list 
        for lidx in range(1,nn.num_internal_layers+1):
            # Find and concatenate parent layers
            plist = get_layer_parents(nn.conn_mat.viewkeys(),lidx)
            parent_layers = [layers[i] for i in plist]
            if self._data_format == 'channels_last':
                input_layer = tf.concat(parent_layers,3)
                num_incoming_filters = input_layer.get_shape().as_list()[-1]
            else:
                input_layer = tf.concat(parent_layers,1)
                num_incoming_filters = input_layer.get_shape().as_list()[1]
            # Add layer to layers-list
            nextLayer = self._get_layers(nn,lidx,num_incoming_filters)
            layers.append(nextLayer(input_layer))

        # Define output layer
        plist = get_layer_parents(nn.conn_mat.viewkeys(),lidx+1) # indices for parents of output layer
        parent_layers = [layers[i] for i in plist] # parent layers of output layer
        return parent_layers

        

        