"""
  Unit tests for nn_examples.py
  -- chenweiwei@ict.ac.cn
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name

import os
# Local imports
from utils.base_test_class import BaseTestClass, execute_tests
from nn.nn_visualise import visualise_list_of_nns, visualise_nn
from opt.nn_opt_utils import get_initial_pool
from nn import nn_examples

from demos.model_convert import ConvNNdataNetCifar10    #test for the network cast
from demos.model_cast import model_cast


from multiprocessing import Process, Manager
import Queue
import time

def tranfor_net_to_NNdataFlow(nn):
  model_cast_cifar10 = ConvNNdataNetCifar10(nn)
  NN_dataflow_work = model_cast_cifar10.forward_pass()
  return NN_dataflow_work

def print_nn_info(nn):
    print("==="*15)
    print("lists of layers and num-units in the neural network architecture")
    for lidx in range(1,nn.num_internal_layers+1):
        layerToPrint = nn.layer_labels[lidx]
        unitsToPrint = nn.num_units_in_each_layer[lidx]
        stridevalToPrint = nn.strides[lidx]
        print('layer-label:%-10s, num-units:%4s, strideval:%4s'%(layerToPrint,unitsToPrint,stridevalToPrint))
    print('==='*15)

# Test cases for model_convert_to_data_flow.py ----------------------------------------------------------
class NNConvertTest(BaseTestClass):
  """ Unit test for some neural network examples. We are just testing for generation. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNConvertTest, self).__init__(*args, **kwargs)
    self.save_dir = '../scratch/unittest_examples'

  def test_initial_pool(self):
    """ Unit test for the VGG_net."""
    self.report('Testing the initial pool. ')
    cnns = get_initial_pool('cnn')
    visualise_list_of_nns(cnns, os.path.join(self.save_dir, 'cnn'))
    NN_PERFORMANCE = []
    for nn in cnns:
      NN_dataflow_work = tranfor_net_to_NNdataFlow(nn)
      manager = Manager()
      return_dict = manager.dict()
      p = Process(target=model_cast,args=(NN_dataflow_work,return_dict,))
      p.start()
      p.join()

      NN_PERFORMANCE.append(return_dict['total_time'])
      print("NN_PERFORMANCE:",NN_PERFORMANCE)

  
'''
 def test_vgg_16(self):
    self.report('Testing the resnet. ')
    vgg_16 = nn_examples.get_resnet_cnn(3, 2, 1)
    print_nn_info(vgg_16)
    save_file_prefix = os.path.join(self.save_dir, "resnet")
    visualise_nn(vgg_16, save_file_prefix)
    NN_dataflow_work = tranfor_net_to_NNdataFlow(vgg_16)
    manager = Manager()
    return_dict = manager.dict()
    p = Process(target=model_cast,args=(NN_dataflow_work,return_dict,))
    p.start()
    p.join()
    print("cost:",return_dict['total_cost'])
'''

if __name__ == '__main__':
  execute_tests()

