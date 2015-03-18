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

# generate_dataset.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import weka.datagenerators as datagen
import wekaexamples.helper as helper
from weka.flow.control import Flow
from weka.flow.source import DataGenerator
from weka.flow.sink import Console


def load_batch():
    """
    Loads a dataset in batch mode.
    """



def load_incremental():
    """
    Loads a dataset incrementally.
    """

    # setup the flow
    helper.print_title("Load dataset (incremental)")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="load dataset")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = True
    flow.actors.append(loaddataset)

    console = Console()
    flow.actors.append(console)

    # run the flow
    msg = flow.setup()
    if msg is None:
        msg = flow.execute()
        if msg is not None:
            print("Error executing flow:\n" + msg)
    else:
        print("Error setting up flow:\n" + msg)
    flow.wrapup()
    flow.cleanup()


def load_custom_loader():
    """
    Loads a dataset using a custom loader.
    """

    # setup the flow
    helper.print_title("Load dataset (custom loader)")
    iris = helper.get_data_dir() + os.sep + "iris.csv"

    flow = Flow(name="load dataset")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = False
    loaddataset.config["use_custom_loader"] = True
    loaddataset.config["custom_loader"] = Loader(classname="weka.core.converters.CSVLoader")
    flow.actors.append(loaddataset)

    console = Console()
    flow.actors.append(console)

    # run the flow
    msg = flow.setup()
    if msg is None:
        msg = flow.execute()
        if msg is not None:
            print("Error executing flow:\n" + msg)
    else:
        print("Error setting up flow:\n" + msg)
    flow.wrapup()
    flow.cleanup()


def main():
    """
    Just runs some example code.
    """
    # setup the flow
    helper.print_title("Generate dataset")

    flow = Flow(name="generate dataset")

    generator = DataGenerator()
    generator.config["setup"] = datagen.DataGenerator(classname="weka.datagenerators.classifiers.classification.Agrawal")
    flow.actors.append(generator)

    console = Console()
    flow.actors.append(console)

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
