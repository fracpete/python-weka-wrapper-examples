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

# attribute_selection.py
# Copyright (C) 2014 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    anneal_file = helper.get_data_dir() + os.sep + "anneal.arff"
    helper.print_info("Loading dataset: " + anneal_file)
    loader = Loader("weka.core.converters.ArffLoader")
    anneal_data = loader.load_file(anneal_file)
    anneal_data.class_is_last()

    # perform attribute selection
    helper.print_title("Attribute selection")
    search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(anneal_data)
    print("# attributes: " + str(attsel.number_attributes_selected))
    print("attributes: " + str(attsel.selected_attributes))
    print("result string:\n" + attsel.results_string)

    # perform ranking
    helper.print_title("Attribute ranking (2-fold CV)")
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation("weka.attributeSelection.InfoGainAttributeEval")
    attsel = AttributeSelection()
    attsel.ranking(True)
    attsel.folds(2)
    attsel.crossvalidation(True)
    attsel.seed(42)
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(anneal_data)
    print("ranked attributes:\n" + str(attsel.ranked_attributes))
    print("result string:\n" + attsel.results_string)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
