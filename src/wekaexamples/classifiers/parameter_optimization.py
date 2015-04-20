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
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.classes import ListParameter, MathParameter
from weka.classifiers import Classifier, GridSearch, MultiSearch


def gridsearch():
    """
    Applies GridSearch to a dataset. GridSearch package must be not be installed, as the monolithic weka.jar
    already contains this package.
    """

    helper.print_title("GridSearch")

    # load a dataset
    fname = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading train: " + fname)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(fname)
    train.class_is_last()

    # classifier
    grid = GridSearch(options=["-sample-size", "100.0", "-traversal", "ROW-WISE", "-num-slots", "1", "-S", "1"])
    grid.evaluation = "CC"
    grid.y = {"property": "kernel.gamma", "min": -3.0, "max": 3.0, "step": 1.0, "base": 10.0, "expression": "pow(BASE,I)"}
    grid.x = {"property": "C", "min": -3.0, "max": 3.0, "step": 1.0, "base": 10.0, "expression": "pow(BASE,I)"}
    cls = Classifier(
        classname="weka.classifiers.functions.SMOreg",
        options=["-K", "weka.classifiers.functions.supportVector.RBFKernel"])
    grid.classifier = cls
    grid.build_classifier(train)
    print("Model:\n" + str(grid))
    print("\nBest setup:\n" + grid.best.to_commandline())


def multisearch():
    """
    Applies MultiSearch to a dataset. "multisearch-weka-package" package must be installed.
    """

    helper.print_title("MultiSearch")

    # load a dataset
    fname = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading train: " + fname)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(fname)
    train.class_is_last()

    # classifier
    multi = MultiSearch(
        options=["-sample-size", "100.0", "-initial-folds", "2", "-subsequent-folds", "2",
                 "-num-slots", "1", "-S", "1"])
    multi.evaluation = "CC"
    mparam = MathParameter()
    mparam.prop = "classifier.kernel.gamma"
    mparam.minimum = -3.0
    mparam.maximum = 3.0
    mparam.step = 1.0
    mparam.base = 10.0
    mparam.expression = "pow(BASE,I)"
    lparam = ListParameter()
    lparam.prop = "classifier.C"
    lparam.values = ["-2.0", "-1.0", "0.0", "1.0", "2.0"]
    multi.parameters = [mparam, lparam]
    cls = Classifier(
        classname="weka.classifiers.functions.SMOreg",
        options=["-K", "weka.classifiers.functions.supportVector.RBFKernel"])
    multi.classifier = cls
    multi.build_classifier(train)
    print("Model:\n" + str(multi))
    print("\nBest setup:\n" + multi.best.to_commandline())


def main():
    """
    Calls the parameter optimization method(s).
    """
    #gridsearch()
    multisearch()


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
