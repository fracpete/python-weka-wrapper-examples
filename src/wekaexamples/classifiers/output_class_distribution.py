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

# output_class_distribution.py
# Copyright (C) 2014 Fracpete (pythonwekawrapper at gmail dot com)

import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier


def main(args):
    """
    Trains a J48 classifier on a training set and outputs the predicted class and class distribution alongside the
    actual class from a test set. Class attribute is assumed to be the last attribute.
    :param args: the commandline arguments (train and test datasets)
    :type args: list
    """

    # load a dataset
    helper.print_info("Loading train: " + args[1])
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(args[1])
    train.class_index = train.num_attributes - 1
    helper.print_info("Loading test: " + args[2])
    test = loader.load_file(args[2])
    test.class_is_last()

    # classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)

    # output predictions
    print("# - actual - predicted - error - distribution")
    for i in xrange(test.num_instances):
        inst = test.get_instance(i)
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        print(
            "%d - %s - %s - %s  - %s" %
            (i+1,
             inst.get_string_value(inst.class_index),
             inst.class_attribute.value(int(pred)),
             "yes" if pred != inst.get_value(inst.class_index) else "no",
             str(dist.tolist())))


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
