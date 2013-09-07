Binarization
============

An implementation of some binarization methods such as Niblack, Sauvola, Wolfjolion [1] and one based on feature space partitioning that uses the others as auxiliary methods [2].

Compile using the Makefile. There is also a Shell Script that makes it possible to run the code with different input images and different binarization methods.

This code is ugly and was made in a hurry. But works pretty well. One of the major needed refactorings is use just one version of OpenCV, and make better usage of space/memory.

Check directories input and output for input images and output images using different binarization methods.

For example, for this input image (http://homes.dcc.ufba.br/~caiosba/mestrado/binarization/d.png), this one (http://homes.dcc.ufba.br/~caiosba/mestrado/binarization/d.this.w.png) is produced as output when using the feature space partitioning algorithm with Wolfjolion as auxiliary method.

The program can be run this way:

`./binary <input image name without extension (it will assume .png)> <binarization method>`

Binarization method can be one of the following: s (Sauvola), n (Niblack) or w (Wolfjolion). Each execution will produce two images: one using the method as primary method and another using it as an auxliary method for the feature space partitioning binarization.

Developed as a work for the master lecture Topics on Visual Computing, by Profs. Vinícius and Perfilino, at Federal University of Bahia - Brazil.

References:

* 1. http://liris.cnrs.fr/christian.wolf/software/binarize/
* 2. http://dx.doi.org/10.1007/s10032-010-0142-4
