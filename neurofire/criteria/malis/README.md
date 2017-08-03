# Malis Loss

## Build

This requires nifty, which can be included as git submodule:

```
$ cd malis_impl
$ git submodule update --init --recursive
```

Next, the malis C++ implementation needs to be built with make and cmake:

```
$ mkdir bld
$ cd bld
$ cmake ..
$ make
```

## Usage

To use Malis, simply: 

```python
from neurofire.criteria.malis import Malis
```
and use `Malis` as the training criteria. 
