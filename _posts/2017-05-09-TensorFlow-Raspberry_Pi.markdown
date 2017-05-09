---
layout: post
title:  "TensorFlow on Raspberry Pi"
date:   2017-05-09 10:37:00
---

In post shares the steps to deploy TensorFlow models on Raspberry Pi

# Install TensorFlow on a Raspberry Pi
First thing first, let's install TensorFlow on a Raspberry Pi. There are several options, follow the [official page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile) using a C++ library or off-the-shelf [project](https://github.com/samjabrahams/tensorflow-on-raspberry-pi) using standard Python. The later option seems to support RPi 3 only.

## Official method
```
git clone https://github.com/tensorflow/tensorflow.git
tensorflow/contrib/makefile/download_dependencies.sh
sudo apt-get update
sudo apt-get -f upgrade
sudo apt-get install -y autoconf automake libtool gcc-4.8 g++-4.8
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure
make
sudo make install
sudo ldconfig  # refresh shared library cache
cd ../../../../..
```

Build the library and example with extra optimization flags to give you code that will run faster on RPi 2, 3 (yay)
```
make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI \
OPTFLAGS="-Os -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize" CXX=g++-4.8
```

Or with other devices:
```
make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI OPTFLAGS="-Os" CXX=g++-4.8
```

One thing to be careful of is that the `gcc` version 4.9 currently installed on *Jessie* by default will hit an error mentioning `__atomic_compare_exchange`. This is why the examples above specify `CXX=g++-4.8` explicitly, and why we install it using `apt-get`. If you have partially built using the default _gcc 4.9_, hit the error and switch to **4.8**, you need to do a `make -f tensorflow/contrib/makefile/Makefile clean` before you build. If you don't, the build will appear to succeed but you'll encounter [malloc(): memory corruption errors](https://github.com/tensorflow/tensorflow/issues/3442) when you try to run any programs using the library.

# Deploy TensorFlow model
## Official examples
For more examples, look at the [tensorflow/contrib/pi_examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/pi_examples) folder in the source tree, which contains code samples aimed at the *Raspberry Pi*.
Here, we play with image classification to validate the installed TensorFlow library

Install libjpeg, so we can load image files:
```
sudo apt-get install -y libjpeg-dev
```

To download the example model you'll need, run these commands:
```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015_stripped.zip \
-o /tmp/inception_dec_2015_stripped.zip
unzip /tmp/inception_dec_2015_stripped.zip \
-d tensorflow/contrib/pi_examples/label_image/data/
```

From the root of the TensorFlow source tree, run the following command to build a basic example.
```
make -f tensorflow/contrib/pi_examples/label_image/Makefile
```

Run the example
```
tensorflow/contrib/pi_examples/label_image/gen/bin/label_image
```
to try out image labeling with the default Grace Hopper image.

You should several lines of output, with "Military Uniform" shown as the top result, something like this:
```
I tensorflow/contrib/pi_examples/label_image/label_image.cc:384] Running model succeeded!
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] military uniform (866): 0.624293
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] suit (794): 0.0473981
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] academic gown (896): 0.0280926
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] bolo tie (940): 0.0156956
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] bearskin (849): 0.0143348
```

Once you've verified that is working, you can supply your own images with `--image=your_image.jpg`, or even with graphs you've trained yourself with the TensorFlow for Poets tutorial using `--graph=your_graph.pb --input=Mul:0 --output=final_result:0`.

Note: here, we use the model ***inception_dec_2015_stripped***
