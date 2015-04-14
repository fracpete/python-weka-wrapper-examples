# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# parameter_optimization.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import sys
import traceback
import weka.core.jvm as jvm
import javabridge
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer


def gridsearch():
    """
    Applies GridSearch to a dataset. GridSearch package must be not be installed, as the monolithic weka.jar
    already contains this package.
    """

    # load a dataset
    fname = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading train: " + fname)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(fname)
    train.class_is_last()

    # classifier
    grid = SingleClassifierEnhancer(
        classname="weka.classifiers.meta.GridSearch",
        options=[
            "-E", "CC",
            "-y-property", "kernel.gamma", "-y-min", "-3.0", "-y-max", "3.0", "-y-step", "1.0", "-y-base", "10.0",
            "-y-expression", "pow(BASE,I)",
            "-x-property", "C", "-x-min", "-3.0", "-x-max", "3.0", "-x-step", "1.0", "-x-base", "10.0",
            "-x-expression", "pow(BASE,I)",
            "-sample-size", "100.0", "-traversal", "ROW-WISE", "-num-slots", "1", "-S", "1"])
    cls = Classifier(
        classname="weka.classifiers.functions.SMOreg",
        options=["-K", "weka.classifiers.functions.supportVector.RBFKernel"])
    grid.classifier = cls
    grid.build_classifier(train)
    print(str(grid))
    best = Classifier(jobject=javabridge.call(grid.jobject, "getBestClassifier", "()Lweka/classifiers/Classifier;"))
    print(best.to_commandline())


def main():
    """
    Calls the parameter optimization method(s).
    """
    gridsearch()


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
