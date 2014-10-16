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

# attribute_selection_test.py
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

import os
import sys
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.core.dataset import Instances, Instance, Attribute
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter


def create_dataset_header():
    """
    Creates the dataset header.
    :return: the header
    :rtype: Instances
    """
    att_msg = Attribute.create_string("Message")
    att_cls = Attribute.create_nominal("Class", ["miss", "hit"])
    result = Instances.create_instances("MessageClassificationProblem", [att_msg, att_cls], 0)
    return result


def main(args):
    """
    TODO
    :param args: the commandline arguments
    :type args: list
    """

    data = create_dataset_header()
    print(str(data))

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception, e:
        print(e)
    finally:
        jvm.stop()
