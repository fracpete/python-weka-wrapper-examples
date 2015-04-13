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

# dump_instances.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
import weka.filters as filters
from weka.flow.control import Flow
from weka.flow.source import FileSupplier
from weka.flow.transformer import LoadDataset, Filter, RenameRelation
from weka.flow.sink import InstanceDumper


def main():
    """
    Just runs some example code.
    """
    """
    Loads/filters a dataset incrementally and saves it to a new file.
    """

    # setup the flow
    helper.print_title("Load/filter/save dataset (incrementally)")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="Load/filter/save dataset (incrementally)")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    flow.actors.append(loaddataset)

    flter = Filter()
    flter.config["setup"] = filters.Filter(
        classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
    flow.actors.append(flter)

    rename = RenameRelation()
    rename.config["name"] = "iris-reduced"
    flow.actors.append(rename)

    dumper = InstanceDumper()
    dumper.config["output"] = tempfile.gettempdir() + os.sep + "out.arff"
    flow.actors.append(dumper)

    # run the flow
    msg = flow.setup()
    if msg is None:
        print("\n" + flow.tree + "\n")
        msg = flow.execute()
        if msg is not None:
            print("Error executing flow:\n" + msg)
    else:
        print("Error setting up flow:\n" + msg)
    flow.wrapup()
    flow.cleanup()

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
