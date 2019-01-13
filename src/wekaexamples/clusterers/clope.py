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

# clope.py
# Copyright (C) 2019 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
from weka.clusterers import Clusterer
from weka.core.packages import install_package, is_installed


def main():
    if not is_installed("CLOPE"):
        print("CLOPE is not installed, installing now")
        install_package("CLOPE")
        print("please restart")
        return

    cls = Clusterer(classname="weka.clusterers.CLOPE")
    print("CLOPE is installed:", cls.to_commandline())


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
