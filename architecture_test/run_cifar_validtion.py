"""
  retrain the best architecture we have serched from scrach.
  -- chenweiwei@ict.ac.cn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os
import time
import argparse
import tensorflow as tf
import cifar_main 
from cifar import cifar10_utils, cifar10

from nn.nn_examples import get_vgg_net

def get_experiment_fn(nnObj,data_dir, num_gpus,variable_strategy,use_distortion_for_training=True):
  """Returns an Experiment function.

  Experiments perform training on several workers in parallel,
  in other words experiments know how to invoke train and eval in a sensible
  fashion for distributed training. Arguments passed directly to this
  function are not tunable, all other arguments should be passed within
  tf.HParams, passed to the enclosed function.

  Args:
      data_dir: str. Location of the data for input_fns.
      num_gpus: int. Number of GPUs on each worker.
      variable_strategy: String. CPU to use CPU as the parameter server
      and GPU to use the GPUs as the parameter server.
      use_distortion_for_training: bool. See cifar10.Cifar10DataSet.
  Returns:
      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
      tf.contrib.learn.Experiment.

      Suitable for use by tf.contrib.learn.learn_runner, which will run various
      methods on Experiment (train, evaluate) based on information
      about the current runner in `run_config`.
  """

  def _experiment_fn(run_config, hparams):
    """Returns an Experiment."""
    # Create estimator.
    train_input_fn = functools.partial(
        cifar_main.input_fn,
        data_dir,
        subset='train',
        num_shards=num_gpus,
        batch_size=hparams.train_batch_size,
        use_distortion_for_training=use_distortion_for_training)

    eval_input_fn = functools.partial(
        cifar_main.input_fn,
        data_dir,
        subset='eval',
        batch_size=hparams.eval_batch_size,
        num_shards=num_gpus)

    num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
    if num_eval_examples % hparams.eval_batch_size != 0:
      raise ValueError(
          'validation set size must be multiple of eval_batch_size')

    train_steps = hparams.train_steps
    eval_steps = num_eval_examples // hparams.eval_batch_size
 
    classifier = tf.estimator.Estimator(
        model_fn=cifar_main.get_model_fn(nnObj,num_gpus, variable_strategy,
                              run_config.num_worker_replicas or 1),
        config=run_config,
        params=hparams)

    vail_accuracy=[]
    for loop in range(20):
      classifier.train(train_input_fn,steps=train_steps)
      vail_accuracy.append(classifier.evaluate(eval_input_fn,steps=eval_steps))
      print("finished iter:"+str((loop+1)*train_steps))
    print("accuracy:")
    print(vail_accuracy)
    # Create experiment.
    return tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        eval_steps=eval_steps)

  return _experiment_fn


def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = cifar10_utils.RunConfig(
      session_config=sess_config, model_dir=job_dir)
  nn = get_vgg_net()
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(nn,data_dir, num_gpus, variable_strategy,
                        use_distortion_for_training),
      run_config=config,
      hparams=tf.contrib.training.HParams(
          is_chief=config.is_chief,
          **hparams))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      default="../data/cifar10",
      type=str,
      help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      default="./experiment",
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='GPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=4000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.')
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.')
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """)
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--data-format',
      type=str,
      default=None,
      help="""\
      If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
  parser.add_argument(
      '--batch-norm-decay',
      type=float,
      default=0.997,
      help='Decay for batch norm.')
  parser.add_argument(
      '--batch-norm-epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.')
  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

  main(**vars(args))
