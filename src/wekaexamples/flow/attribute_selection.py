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

# attribute_selection.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
from weka.attribute_selection import ASSearch, ASEvaluation
import wekaexamples.helper as helper
from weka.flow.control import Flow, ContainerValuePicker, Tee
from weka.flow.source import FileSupplier
from weka.flow.transformer import LoadDataset, AttributeSelection
from weka.flow.sink import Console


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    helper.print_title("Attribute selection")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="attribute selection")

    filesupplier = FileSupplier()
    filesupplier.config["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    loaddataset.config["incremental"] = False
    flow.actors.append(loaddataset)

    attsel = AttributeSelection()
    attsel.config["search"] = ASSearch(classname="weka.attributeSelection.BestFirst")
    attsel.config["eval"] = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
    flow.actors.append(attsel)

    results = Tee()
    results.name = "output results"
    flow.actors.append(results)

    picker = ContainerValuePicker()
    picker.config["value"] = "Results"
    picker.config["switch"] = True
    results.actors.append(picker)

    console = Console()
    console.config["prefix"] = "Attribute selection results:"
    results.actors.append(console)

    reduced = Tee()
    reduced.name = "reduced dataset"
    flow.actors.append(reduced)

    picker = ContainerValuePicker()
    picker.config["value"] = "Reduced"
    picker.config["switch"] = True
    reduced.actors.append(picker)

    console = Console()
    console.config["prefix"] = "Reduced dataset:\n\n"
    reduced.actors.append(console)

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
