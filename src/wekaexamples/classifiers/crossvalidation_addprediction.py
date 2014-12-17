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

# crossvalidation_addprediction.py
# Copyright (C) 2014 Fracpete (pythonwekawrapper at gmail dot com)

import os
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    data_file = helper.get_data_dir() + os.sep + "vote.arff"
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.set_class_index(data.num_attributes() - 1)

    # classifier
    classifier = Classifier(classname="weka.classifiers.trees.J48", options=None)

    # randomize data
    folds = 10
    seed = 1
    rnd = Random(seed)
    rand_data = Instances.copy_instances(data)
    rand_data.randomize(rnd)
    if rand_data.get_class_attribute().is_nominal():
        rand_data.stratify(folds)

    # perform cross-validation and add predictions
    predicted_data = None
    evaluation = Evaluation(rand_data)
    for i in xrange(folds):
        train = rand_data.train_cv(folds, i)
        # the above code is used by the StratifiedRemoveFolds filter,
        # the following code is used by the Explorer/Experimenter
        # train = rand_data.train_cv(folds, i, rnd)
        test = rand_data.test_cv(folds, i)

        # build and evaluate classifier
        cls = Classifier.make_copy(classifier)
        cls.build_classifier(train)
        evaluation.test_model(cls, test)

        # add predictions
        addcls = Filter(
            classname="weka.filters.supervised.attribute.AddClassification",
            options=["-classification", "-distribution", "-error"])
        # setting the java object directory avoids issues with correct quoting in option array
        addcls.set_property("classifier", Classifier.make_copy(classifier))
        addcls.set_inputformat(train)
        addcls.filter(train)  # trains the classifier
        pred = addcls.filter(test)
        if predicted_data is None:
            predicted_data = Instances.template_instances(pred, 0)
        for n in xrange(pred.num_instances()):
            predicted_data.add_instance(pred.get_instance(n))

    print("")
    print("=== Setup ===")
    print("Classifier: " + cls.to_commandline())
    print("Dataset: " + data.get_relationname())
    print("Folds: " + str(folds))
    print("Seed: " + str(seed))
    print("")
    print(evaluation.to_summary("=== " + str(folds) + " -fold Cross-Validation ==="))
    print("")
    print(predicted_data)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(e)
    finally:
        jvm.stop()
