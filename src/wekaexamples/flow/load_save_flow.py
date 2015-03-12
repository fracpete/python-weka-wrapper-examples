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

# load_save_flow.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import tempfile
import weka.core.jvm as jvm
from weka.flow.control import Flow
from weka.flow.source import ListFiles
from weka.flow.sink import Console


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="list files")

    listfiles = ListFiles()
    listfiles.options["dir"] = str(tempfile.gettempdir())
    listfiles.options["list_files"] = True
    listfiles.options["list_dirs"] = False
    listfiles.options["recursive"] = False
    listfiles.options["regexp"] = ".*r.*"
    flow.actors.append(listfiles)

    console = Console()
    console.options["prefix"] = "Match: "
    flow.actors.append(console)

    # print flow
    flow.setup()
    print(flow.tree)

    # save the flow
    fname = tempfile.gettempdir() + os.sep + "simpleflow.json"
    Flow.save(flow, fname)

    # load flow
    flow = Flow.load(fname)

    # output flow
    print(flow)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
