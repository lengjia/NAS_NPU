# nas for processor-aware by BO and OPT

This is a python implemention of searching toward pareto-optimal processor-aware network architecture search by BO and optimal transport.

This repo in based on ['NASBOT'](https://github.com/kirthevasank/nasbot.git), and is changed for processor aware nureal architecture search. by this way, the opensource neural network accelerator Eyeriss is use for hardware test. we mainly focus on the accuacy and energy consumpation.

For more details, please see our paper below.

For questions and bug reports please email chenweiwei@ict.ac.cn.

### Installation

* Download the package.
```bash
$ git clone https://github.com/kirthevasank/nasbot.git
```

* Install the following packages packages via pip: cython, POT (Python Optimal Transport),
graphviz and pygraphviz. graphviz and pygraphviz are only needed to visualise the networks
and are not necessary to run nasbot. However, some unit tests may fail.
```bash
$ pip install cython POT graphviz pygraphviz
```
  In addition to the above, you will need numpy and scipy which can also be pip installed.

* Now set `HOME_PATH` in the set_up file to the parent directory of nasbot, i.e.
`HOME_PATH=<path/to/parent/directory>/nasbot`. Then source the set up file.
```bash
$ source set_up
```

* Next, you need to build the direct fortran library. For this `cd` into
[`utils/direct_fortran`](https://github.com/kirthevasank/nasbot/blob/master/utils/direct_fortran)
and run `bash make_direct.sh`. You will need a fortran compiler such as gnu95.
Once this is done, you can run `python simple_direct_test.py` to make sure that it was
installed correctly.
The default version of NASBOT can be run without direct, but some unit tests might fail.

* Finally, you need to install tensorflow to execute the MLP/CNN demos on GPUs.
```bash
$ pip install tensorflow-gpu
```

**Testing the Installation**:
To test the installation, run ```bash run_all_tests.sh```. Some of the tests are
probabilistic and could fail at times. If this happens, run the same test several times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simpy via `python unittest_xxx.py`.

### Getting started



### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt].

"Copyright 2018 chenweiwei@ict.ac.cn"

- For questions and bug reports please email chenweiwei@ict.ac.cn

