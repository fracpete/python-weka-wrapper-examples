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

# init_storage_value.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
from weka.flow.control import Flow, Trigger
from weka.flow.source import ForLoop, Start
from weka.flow.sink import Console
from weka.flow.transformer import InitStorageValue


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="init storage value")

    start = Start()
    flow.actors.append(start)

    init = InitStorageValue()
    init.options["storage_name"] = "max"
    init.options["value"] = "int(3)"
    flow.actors.append(init)

    trigger = Trigger()
    flow.actors.append(trigger)

    inner = ForLoop()
    inner.name = "inner"
    inner.options["max"] = "@{max}"
    trigger.actors.append(inner)

    console = Console()
    trigger.actors.append(console)

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
