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

# classify_data.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import tempfile
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.classifiers import Classifier
from weka.flow.control import Flow, ContainerValuePicker, Trigger
from weka.flow.source import FileSupplier, Start
from weka.flow.transformer import LoadDataset, ClassSelector, Train, SetStorageValue, Predict
from weka.flow.sink import Console, ModelWriter


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    helper.print_title("classify data")
    iris = helper.get_data_dir() + os.sep + "iris.arff"
    clsfile = str(tempfile.gettempdir()) + os.sep + "j48.model"

    flow = Flow(name="classify data")

    start = Start()
    flow.actors.append(start)

    build_save = Trigger()
    build_save.name = "build and save classifier"
    flow.actors.append(build_save)

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    build_save.actors.append(filesupplier)

    loaddataset = LoadDataset()
    build_save.actors.append(loaddataset)

    select = ClassSelector()
    select.config["index"] = "last"
    build_save.actors.append(select)

    ssv = SetStorageValue()
    ssv.config["storage_name"] = "data"
    build_save.actors.append(ssv)

    train = Train()
    train.config["setup"] = Classifier(classname="weka.classifiers.trees.J48")
    build_save.actors.append(train)

    ssv = SetStorageValue()
    ssv.config["storage_name"] = "model"
    build_save.actors.append(ssv)

    pick = ContainerValuePicker()
    pick.config["value"] = "Model"
    build_save.actors.append(pick)

    console = Console()
    console.config["prefix"] = "built: "
    pick.actors.append(console)

    writer = ModelWriter()
    writer.config["output"] = clsfile
    build_save.actors.append(writer)

    pred_serialized = Trigger()
    pred_serialized.name = "make predictions (serialized model)"
    flow.actors.append(pred_serialized)

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    pred_serialized.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    pred_serialized.actors.append(loaddataset)

    select = ClassSelector()
    select.config["index"] = "last"
    pred_serialized.actors.append(select)

    predict = Predict()
    predict.config["model"] = clsfile
    pred_serialized.actors.append(predict)

    console = Console()
    console.config["prefix"] = "serialized: "
    pred_serialized.actors.append(console)

    pred_storage = Trigger()
    pred_storage.name = "make predictions (model from storage)"
    flow.actors.append(pred_storage)

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    pred_storage.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    pred_storage.actors.append(loaddataset)

    select = ClassSelector()
    select.config["index"] = "last"
    pred_storage.actors.append(select)

    predict = Predict()
    predict.config["storage_name"] = "model"
    pred_storage.actors.append(predict)

    console = Console()
    console.config["prefix"] = "storage: "
    pred_storage.actors.append(console)

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
