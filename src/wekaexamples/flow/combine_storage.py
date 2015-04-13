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

# combine_storage.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
from weka.flow.control import Flow, Trigger
from weka.flow.source import ForLoop, CombineStorage
from weka.flow.sink import Console
from weka.flow.transformer import SetStorageValue


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="combine storage")

    outer = ForLoop()
    outer.name = "outer"
    outer.config["max"] = 3
    flow.actors.append(outer)

    ssv = SetStorageValue()
    ssv.config["storage_name"] = "max"
    flow.actors.append(ssv)

    trigger = Trigger()
    flow.actors.append(trigger)

    inner = ForLoop()
    inner.name = "inner"
    inner.config["max"] = "@{max}"
    trigger.actors.append(inner)

    ssv2 = SetStorageValue()
    ssv2.config["storage_name"] = "inner"
    trigger.actors.append(ssv2)

    trigger2 = Trigger()
    trigger.actors.append(trigger2)

    combine = CombineStorage()
    combine.config["format"] = "@{max} / @{inner}"
    trigger2.actors.append(combine)

    console = Console()
    trigger2.actors.append(console)

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
