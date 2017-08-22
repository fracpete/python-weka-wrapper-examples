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

# mp5.py
# Copyright (C) 2017 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    bodyfat_file = helper.get_data_dir() + os.sep + "bodyfat.arff"
    helper.print_info("Loading dataset: " + bodyfat_file)
    loader = Loader("weka.core.converters.ArffLoader")
    bodyfat_data = loader.load_file(bodyfat_file)
    bodyfat_data.class_is_last()

    # classifier help
    helper.print_title("Creating help string")
    classifier = Classifier(classname="weka.classifiers.trees.M5P")
    classifier.build_classifier(bodyfat_data)
    print(classifier)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
