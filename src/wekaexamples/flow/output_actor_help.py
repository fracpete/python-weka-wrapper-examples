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

# output_actor_help.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.classifiers import Classifier
from weka.flow.control import Flow
from weka.flow.source import FileSupplier
from weka.flow.transformer import LoadDataset, ClassSelector, CrossValidate
from weka.flow.sink import Console


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    helper.print_title("Output actor help")

    cv = CrossValidate()
    cv.options["setup"] = Classifier(classname="weka.classifiers.trees.J48")
    cv.print_help()

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
