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
# Copyright (C) 2015-2019 Fracpete (pythonwekawrapper at gmail dot com)

import os
import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation, PredictionOutput


def main(args):
    """
    Loads a dataset, shuffles it, splits it into train/test set. Trains J48 with training set and
    evaluates the built model on the test set.
    The predictions get recorded in two different ways:
    1. in-memory via the test_model method
    2. directly to file (more memory efficient), but a separate run of making predictions

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

    # evaluate and record predictions in memory
    helper.print_title("recording predictions in-memory")
    output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-distribution"])
    evl = Evaluation(train)
    evl.test_model(cls, test, output=output)
    print(evl.summary())
    helper.print_info("Predictions:")
    print(output.buffer_content())

    # record/output predictions separately
    helper.print_title("recording/outputting predictions separately")
    outputfile = helper.get_tmp_dir() + "/j48_vote.csv"
    output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-distribution", "-suppress", "-file", outputfile])
    output.header = test
    output.print_all(cls, test)
    helper.print_info("Predictions stored in: " + outputfile)
    # by using "-suppress" we don't store the output in memory, the following statement won't output anything
    print(output.buffer_content())


if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
