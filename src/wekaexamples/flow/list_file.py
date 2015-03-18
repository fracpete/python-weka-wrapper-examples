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

# list_files.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

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
    # flow.print_help()

    listfiles = ListFiles()
    listfiles.config["dir"] = str(tempfile.gettempdir())
    listfiles.config["list_files"] = True
    listfiles.config["list_dirs"] = False
    listfiles.config["recursive"] = False
    listfiles.config["regexp"] = ".*r.*"
    # listfiles.print_help()
    flow.actors.append(listfiles)

    console = Console()
    console.config["prefix"] = "Match: "
    # console.print_help()
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
