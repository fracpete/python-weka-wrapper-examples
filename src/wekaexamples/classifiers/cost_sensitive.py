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

# cost_sensitive.py
# Copyright (C) 2014 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import SingleClassifierEnhancer, Classifier, Evaluation


def main():
    """
    Shows how to use the CostSensitiveClassifier.
    """

    # load a dataset
    data_file = helper.get_data_dir() + os.sep + "diabetes.arff"
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    # classifier
    classifier = SingleClassifierEnhancer(
        classname="weka.classifiers.meta.CostSensitiveClassifier",
        options=["-cost-matrix", "[0 1; 2 0]", "-S", "2"])
    base = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
    classifier.classifier = base

    folds = 10
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(classifier, data, folds, Random(1))


    print("")
    print("=== Setup ===")
    print("Classifier: " + classifier.to_commandline())
    print("Dataset: " + data.relationname)
    print("")
    print(evaluation.summary("=== " + str(folds) + " -fold Cross-Validation ==="))

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
