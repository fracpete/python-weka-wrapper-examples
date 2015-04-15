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

# conversions.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.flow.conversion import AnyToCommandline, CommandlineToAny
from weka.flow.control import Flow
from weka.flow.source import StringConstants
from weka.flow.transformer import Convert
from weka.flow.sink import Console


def main():
    """
    Just runs some example code.
    """
    """
    Tests some conversions.
    """

    # setup the flow
    helper.print_title("conversions")

    flow = Flow(name="conversions")

    strings = StringConstants()
    strings.config["strings"] = ["weka.classifiers.trees.J48", "weka.classifiers.functions.SMO"]
    flow.actors.append(strings)

    c2a = CommandlineToAny()
    c2a.config["wrapper"] = "weka.classifiers.Classifier"
    convert1 = Convert()
    convert1.config["setup"] = c2a
    flow.actors.append(convert1)

    convert2 = Convert()
    convert2.config["setup"] = AnyToCommandline()
    flow.actors.append(convert2)

    console = Console()
    console.config["prefix"] = "setup: "
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
