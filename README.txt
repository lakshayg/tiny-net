 _   _                          _   
| |_(_)_ __  _   _   _ __   ___| |_ 
| __| | '_ \| | | | | '_ \ / _ \ __|
| |_| | | | | |_| | | | | |  __/ |_ 
 \__|_|_| |_|\__, | |_| |_|\___|\__|
Tiny  neural |___/ network in Python

This repository contains a minimal implementation of commonly
used neural networks. The networks can be constructed using
various layers as the building blocks. It allows for choosing
between different layers, activation functions, objective
functions to optimize and different optimization algorithms.
API documentation, examples and other relevant material is
available at https://github.com/lakshayg/tiny-net/wiki

Installation
------------
TinyNet is not packaged as a python package at the moment and
therefore cannot be installed in the traditional sense. To use
the package, clone this repository to some path for example:
`/path/to/repo`. To import TinyNet in your program, you must
add the following two lines at the start of the program

```
import sys
sys.path.append('/path/to/repo/')
```
