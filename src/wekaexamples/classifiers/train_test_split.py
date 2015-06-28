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

# train_test_split.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation


def main(args):
    """
    Loads a dataset, shuffles it, splits it into train/test set. Trains J48 with training set and
    evaluates the built model on the test set.
    :param args: the commandline arguments (optional, can be dataset filename)
    :type args: list
    """

    # load a dataset
    if len(args) <= 1:
        data_file = helper.get_data_dir() + os.sep + "vote.arff"
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    # generate train/test split of randomized data
    train, test = data.train_test_split(66.0, Random(1))

    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
