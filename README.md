# LNS2-MARL

The code requires the external libraries BOOST (https://www.boost.org/) and Eigen (https://eigen.tuxfamily.org/). Here is an easy way of installing the required libraries on Ubuntu:
```
sudo apt update
```

   * Install the Eigen library (used for linear algebra computing)
```
    sudo apt install libeigen3-dev
```
   * Install the boost library
```
    sudo apt install libboost-all-dev
```
After you installed both libraries and downloaded the source code, go into the directory of the source code and compile it with CMake:
```
cmake -DCMAKE_BUILD_TYPE=RELEASE .
make
```
