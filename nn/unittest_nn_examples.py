"""
  Unit tests for nn_examples.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name

import os
# Local imports
from utils.base_test_class import BaseTestClass, execute_tests
from nn import nn_examples
from nn.nn_visualise import visualise_list_of_nns, visualise_nn
# from demos.model_convert import ConvNNdataNetCifar10    #test for the network cast
# from demos.model_cast import model_cast

# from multiprocessing import Process, Manager
# import Queue
# import time

def for_test():
  x = 12
  for i in range(200000000):
    x = x^1 

def get_vgg(vgg_net_chen,num,return_dict):
  model_cast_cifar10 = ConvNNdataNetCifar10(vgg_net_chen)
  NN_dataflow_work = model_cast_cifar10.forward_pass()
  cast_map = model_cast(NN_dataflow_work)
  return_dict[num] = cast_map


# Test cases for nn_examples.py ----------------------------------------------------------
class NNExamplesTestCase(BaseTestClass):
  """ Unit test for some neural network examples. We are just testing for generation. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNExamplesTestCase, self).__init__(*args, **kwargs)
    self.save_dir = '../scratch/unittest_examples'

  def test_vgg(self):
    """ Unit test for the VGG_net."""
    self.report('Testing the VGG net. ')
    vggnet2 = nn_examples.get_vgg_net(3)
    save_file_prefix = os.path.join(self.save_dir, "vgg-16")
    visualise_nn(vggnet2, save_file_prefix)

  # def test_vgg_chen(self):
  #   self.report('Testing the vgg_net_chen. ')
  #   NN_PERFORMANCE = []

  #   vgg_net_chen = nn_examples.get_vgg_net_chen(3)
  #   save_file_prefix = os.path.join(self.save_dir, "vgg-chen")
  #   manager = Manager()
  #   return_dict = manager.dict()
  #   model_cast_cifar10 = ConvNNdataNetCifar10(vgg_net_chen)
  #   NN_dataflow_work = model_cast_cifar10.forward_pass()
    
  #   p = Process(target=model_cast,args=(NN_dataflow_work,return_dict))
  #   p.start()
  #   t1 = time.time()
  #   print("time-1:",t1)
  #   for_test()
  #   t2 = time.time()
  #   print("time-2:",t2-t1)
  #   p.join()
  #   NN_PERFORMANCE.append(return_dict['total_cost'])
  #   t3 = time.time()
  #   print("time-3:",t3-t1)
  #   with open("../result/performance.txt",'w+') as f:
  #         f.write(str(NN_PERFORMANCE))
  #         f.write("\n")
  #         f.close()
    #print(return_dict)

    #visualise_nn(vgg_net_chen, save_file_prefix)
'''
  def test_cnn(self):
    self.report('Testing the cnn net. ')
    cnn_net = nn_examples.generate_cnn_architectures()
    save_file_prefix = os.path.join(self.save_dir, "3")
    visualise_list_of_nns(cnn_net, save_file_prefix)

    
  def test_blocked_cnn(self):
    """ Unit test for a blocked CNN. """
    self.report('Testing a blocked CNN.')
    blocknet4 = nn_examples.get_blocked_cnn(4, 4, 1)
    save_file_prefix = os.path.join(self.save_dir, "2")
    visualise_nn(blocknet4, save_file_prefix)

  def test_generate_many_nns(self):
    """ Testing generation of many neural networks. """
    self.report('Testing generation of many NNs.')
    num_nns = 10
    cnns = nn_examples.generate_many_neural_networks('cnn', num_nns)
    visualise_list_of_nns(cnns, os.path.join(self.save_dir, 'cnn'))
    reg_mlps = nn_examples.generate_many_neural_networks('mlp-reg', num_nns)
    visualise_list_of_nns(reg_mlps, os.path.join(self.save_dir, 'reg_mlps'))
    class_mlps = nn_examples.generate_many_neural_networks('mlp-class', num_nns)
    visualise_list_of_nns(class_mlps, os.path.join(self.save_dir, 'class_mlps'))

'''
if __name__ == '__main__':
  execute_tests()

