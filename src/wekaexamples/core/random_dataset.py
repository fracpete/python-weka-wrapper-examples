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

# random_dataset.py
# Copyright (C) 2015 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances


def main():
    """
    Creates a dataset from scratch using random data and outputs it.
    """

    atts = []
    for i in xrange(5):
        atts.append(Attribute.create_numeric("x" + str(i)))

    data = Instances.create_instances("data", atts, 10)

    for n in xrange(10):
        values = []
        for i in xrange(5):
            values.append(n*100 + i)
        inst = Instance.create_instance(values)
        data.add_instance(inst)

    print(data)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
