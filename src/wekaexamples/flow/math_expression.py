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

# math_expression.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
from weka.flow.control import Flow, Trigger
from weka.flow.source import ForLoop
from weka.flow.sink import Console
from weka.flow.transformer import MathExpression


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="math expression")

    outer = ForLoop()
    outer.options["max"] = 100
    flow.actors.append(outer)

    expr = MathExpression()
    expr.options["expression"] = "math.sqrt(X)"
    flow.actors.append(expr)

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
