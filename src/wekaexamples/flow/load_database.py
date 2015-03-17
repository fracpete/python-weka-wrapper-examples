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

# load_database.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.flow.control import Flow
from weka.flow.source import LoadDatabase
from weka.flow.sink import Console


def main():
    """
    Just runs some example code.
    """
    """
    Loads data from a database.
    """

    # setup the flow
    helper.print_title("Load from database")

    flow = Flow(name="load from database")

    loaddatabase = LoadDatabase()
    loaddatabase.options["db_url"] = "jdbc:mysql://HOSTNAME:3306/DBNAME"
    loaddatabase.options["user"] = "DBUSER"
    loaddatabase.options["password"] = "DBPW"
    loaddatabase.options["query"] = "select * from TABLE"
    flow.actors.append(loaddatabase)

    console = Console()
    flow.actors.append(console)

    # run the flow
    msg = flow.setup()
    if msg is None:
        print(flow.tree)
        msg = flow.execute()
        if msg is not None:
            print("Error executing flow:\n" + msg)
    else:
        print("Error setting up flow:\n" + msg)
    flow.wrapup()
    flow.cleanup()

if __name__ == "__main__":
    try:
        mysql_jar = "/some/where/mysql-connector-java-X.Y.Z-bin.jar"
        jvm.start(class_path=[mysql_jar])
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
