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

# database.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.database import InstanceQuery


def main():
    """
    Just runs some example code.
    NB: You have to fill in the following parameters to make it work with MySQL:
    - HOSTNAME: the database server IP or hostname (or just 'localhost')
    - DBNAME: the name of the database to connect to
    - DBUSER: the user for connecting to the database
    - DBPW: the password for the database user
    - TABLE: the table to retrieve the data from
    And also supply the correct path to the MySQL jar in the "main" method below.
    """

    # retrieve some data
    helper.print_title("Loading data from a database")
    iquery = InstanceQuery()
    iquery.db_url = "jdbc:mysql://HOSTNAME:3306/DBNAME"
    iquery.user = "DBUSER"
    iquery.password = "DBPW"
    iquery.query = "select * from TABLE"
    data = iquery.retrieve_instances()
    print(data)

if __name__ == "__main__":
    try:
        mysql_jar = "/some/where/mysql-connector-java-X.Y.Z-bin.jar"
        jvm.start(class_path=[mysql_jar])
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
