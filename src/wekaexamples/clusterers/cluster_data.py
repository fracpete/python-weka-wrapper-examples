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

# cluster_data.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.clusterers import Clusterer, FilteredClusterer, ClusterEvaluation
from weka.filters import Filter
import weka.plot.graph as plg
import weka.plot.clusterers as plc


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(iris_file)

    # remove class attribute
    data.delete_last_attribute()

    # build a clusterer and output model
    helper.print_title("Training SimpleKMeans clusterer")
    clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
    clusterer.build_clusterer(data)
    print(clusterer)

    # cluster data
    helper.print_info("Clustering data")
    for index, inst in enumerate(data):
        cl = clusterer.cluster_instance(inst)
        dist = clusterer.distribution_for_instance(inst)
        print(str(index+1) + ": cluster=" + str(cl) + ", distribution=" + str(dist))

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
