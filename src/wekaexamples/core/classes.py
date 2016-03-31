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

# classes.py
# Copyright (C) 2014 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.classes import Random, SingleIndex, Range, Tag, SelectedTag, JavaObject


def main():
    """
    Just runs some example code.
    """

    # generic JavaObject stuff
    helper.print_title("Generic stuff using weka.core.SystemInfo")
    info = JavaObject(JavaObject.new_instance(classname="weka.core.SystemInfo"))
    jwrapper = info.jwrapper
    print("toString() method:")
    print(jwrapper.toString())

    # random
    helper.print_title("Random")
    rnd = Random(1)
    for i in xrange(10):
        print(rnd.next_double())
    for i in xrange(10):
        print(rnd.next_int(100))

    # single index
    helper.print_title("SingleIndex")
    si = SingleIndex(index="first")
    upper = 10
    si.upper(upper)
    print(str(si) + " (upper=" + str(upper) + ")\n -> " + str(si.index()))
    si.single_index = "3"
    si.upper(upper)
    print(str(si) + " (upper=" + str(upper) + ")\n -> " + str(si.index()))
    si.single_index = "last"
    si.upper(upper)
    print(str(si) + " (upper=" + str(upper) + ")\n -> " + str(si.index()))

    # range
    helper.print_title("Range")
    rng = Range(ranges="first")
    upper = 10
    invert = False
    rng.upper(upper)
    rng.invert = invert
    print(str(rng.ranges) + " (upper=" + str(upper) + ", invert=" + str(invert) + ")\n -> " + str(rng.selection()))
    rng.ranges = "3"
    rng.upper(upper)
    rng.invert = invert
    print(str(rng.ranges) + " (upper=" + str(upper) + ", invert=" + str(invert) + ")\n -> " + str(rng.selection()))
    rng.ranges = "last"
    rng.upper(upper)
    rng.invert = invert
    print(str(rng.ranges) + " (upper=" + str(upper) + ", invert=" + str(invert) + ")\n -> " + str(rng.selection()))
    rng.ranges = "first-last"
    rng.upper(upper)
    rng.invert = invert
    print(str(rng.ranges) + " (upper=" + str(upper) + ", invert=" + str(invert) + ")\n -> " + str(rng.selection()))
    rng.ranges = "3,4,7-last"
    rng.upper(upper)
    rng.invert = invert
    print(str(rng.ranges) + " (upper=" + str(upper) + ", invert=" + str(invert) + ")\n -> " + str(rng.selection()))
    rng.ranges = "3,4,7-last"
    rng.upper(upper)
    invert = True
    rng.invert = invert
    print(str(rng.ranges) + " (upper=" + str(upper) + ", invert=" + str(invert) + ")\n -> " + str(rng.selection()))

    # tag
    helper.print_title("Tag")
    tag = Tag(ident=1, ident_str="one")
    print("tag=" + str(tag) + ", ident=" + str(tag.ident) + ", readable=" + tag.readable)
    tag.ident = 3
    print("tag=" + str(tag) + ", ident=" + str(tag.ident) + ", readable=" + tag.readable)
    tag = Tag(ident=2, ident_str="two", readable="2nd tag")
    print("tag=" + str(tag) + ", ident=" + str(tag.ident) + ", readable=" + tag.readable)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
