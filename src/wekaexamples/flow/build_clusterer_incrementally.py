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

# build_clusterer_incrementally.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.clusterers import Clusterer
import weka.filters as filters
from weka.flow.control import Flow, ContainerValuePicker, Tee, Trigger
from weka.flow.source import FileSupplier, GetStorageValue
from weka.flow.transformer import LoadDataset, ClassSelector, Train, InitStorageValue, UpdateStorageValue, Filter
from weka.flow.sink import Console


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    count = 50
    helper.print_title("build clusterer incrementally")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="build clusterer incrementally")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    initcounter = InitStorageValue()
    initcounter.config["storage_name"] = "counter"
    initcounter.config["value"] = 0
    flow.actors.append(initcounter)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    flow.actors.append(loaddataset)

    remove = Filter(name="remove class attribute")
    remove.config["setup"] = filters.Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
    flow.actors.append(remove)

    inccounter = UpdateStorageValue()
    inccounter.config["storage_name"] = "counter"
    inccounter.config["expression"] = "{X} + 1"
    flow.actors.append(inccounter)

    train = Train()
    train.config["setup"] = Clusterer(classname="weka.clusterers.Cobweb")
    flow.actors.append(train)

    pick = ContainerValuePicker()
    pick.config["value"] = "Model"
    pick.config["switch"] = True
    flow.actors.append(pick)

    tee = Tee(name="output model every " + str(count) + " instances")
    tee.config["condition"] = "@{counter} % " + str(count) + " == 0"
    flow.actors.append(tee)

    trigger = Trigger(name="output # of instances")
    tee.actors.append(trigger)

    getcounter = GetStorageValue()
    getcounter.config["storage_name"] = "counter"
    trigger.actors.append(getcounter)

    console = Console()
    console.config["prefix"] = "# of instances: "
    trigger.actors.append(console)

    console = Console(name="output model")
    tee.actors.append(console)

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
