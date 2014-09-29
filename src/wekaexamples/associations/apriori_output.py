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

# apriori_output.py
# Copyright (C) 2014 Fracpete (fracpete at gmail dot com)

import os
import sys
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.associations import Associator


def main(args):
    """
    Trains Apriori on the specified dataset (uses vote UCI dataset if no dataset specified).
    :param args: the commandline arguments
    :type args: list
    """

    # load a dataset
    if len(args) <= 1:
        data_file = helper.get_data_dir() + os.sep + "vote.arff"
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.set_class_index(data.num_attributes() - 1)

    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori", options=["-c", "-1"])
    apriori.build_associations(data)
    print(str(apriori))

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception, e:
        print(e)
    finally:
        jvm.stop()
