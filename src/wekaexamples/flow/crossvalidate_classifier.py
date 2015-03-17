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

# crossvalidate_classifier.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.classifiers import Classifier
from weka.flow.control import Flow, Branch, Sequence
from weka.flow.source import FileSupplier
from weka.flow.transformer import LoadDataset, ClassSelector, CrossValidate, EvaluationSummary
from weka.flow.sink import Console, ClassifierErrors


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    helper.print_title("Cross-validate classifier")
    iris = helper.get_data_dir() + os.sep + "iris.arff"

    flow = Flow(name="cross-validate classifier")

    filesupplier = FileSupplier()
    filesupplier.options["files"] = [iris]
    flow.actors.append(filesupplier)

    loaddataset = LoadDataset()
    flow.actors.append(loaddataset)

    select = ClassSelector()
    select.options["index"] = "last"
    flow.actors.append(select)

    cv = CrossValidate()
    cv.options["setup"] = Classifier(classname="weka.classifiers.trees.J48")
    flow.actors.append(cv)

    branch = Branch()
    flow.actors.append(branch)

    seqsum = Sequence()
    branch.actors.append(seqsum)

    summary = EvaluationSummary()
    summary.options["title"] = "=== J48/iris ==="
    summary.options["complexity"] = False
    summary.options["matrix"] = True
    seqsum.actors.append(summary)

    console = Console()
    seqsum.actors.append(console)

    seqerr = Sequence()
    branch.actors.append(seqerr)

    errors = ClassifierErrors()
    seqerr.actors.append(errors)

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
