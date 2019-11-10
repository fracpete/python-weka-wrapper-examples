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

# serialization.py
# Copyright (C) 2014-2019 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback
import javabridge
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier
import weka.core.serialization as serialization


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()

    # train classifier
    classifier = Classifier("weka.classifiers.trees.J48")
    classifier.build_classifier(iris_data)

    # save and read object
    helper.print_title("I/O: model (using serialization module)")
    outfile = tempfile.gettempdir() + os.sep + "j48.model"
    serialization.write(outfile, classifier)
    model = Classifier(jobject=serialization.read(outfile))
    print(model)

    # save classifier and dataset header (multiple objects)
    helper.print_title("I/O: model and header (using serialization module)")
    serialization.write_all(outfile, [classifier, Instances.template_instances(iris_data)])
    objects = serialization.read_all(outfile)
    for i, obj in enumerate(objects):
        helper.print_info("Object #" + str(i+1) + ":")
        if javabridge.get_env().is_instance_of(obj, javabridge.get_env().find_class("weka/core/Instances")):
            obj = Instances(jobject=obj)
        elif javabridge.get_env().is_instance_of(obj, javabridge.get_env().find_class("weka/classifiers/Classifier")):
            obj = Classifier(jobject=obj)
        print(obj)

    # save and read object
    helper.print_title("I/O: just model (using Classifier class)")
    outfile = tempfile.gettempdir() + os.sep + "j48.model"
    classifier.serialize(outfile)
    model, _ = Classifier.deserialize(outfile)
    print(model)

    # save classifier and dataset header (multiple objects)
    helper.print_title("I/O: model and header (using Classifier class)")
    classifier.serialize(outfile, header=iris_data)
    model, header = Classifier.deserialize(outfile)
    print(model)
    if header is not None:
        print(header)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
