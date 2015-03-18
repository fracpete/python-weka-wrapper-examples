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

# converters.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import os
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader, TextDirectoryLoader


def main():
    """
    Just runs some example code.
    """

    # load ARFF file
    helper.print_title("Loading ARFF file")
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(helper.get_data_dir() + os.sep + "iris.arff")
    print(str(data))

    # load CSV file
    helper.print_title("Loading CSV file")
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(helper.get_data_dir() + os.sep + "iris.csv")
    print(str(data))

    # load directory
    # changes this to something sensible
    text_dir = "/some/where"
    if os.path.exists(text_dir) and os.path.isdir(text_dir):
        helper.print_title("Loading directory: " + text_dir)
        loader = TextDirectoryLoader(options=["-dir", text_dir, "-F", "-charset", "UTF-8"])
        data = loader.load()
        print(unicode(data))

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
