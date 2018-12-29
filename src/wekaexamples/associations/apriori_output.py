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

# apriori_output.py
# Copyright (C) 2014-2018 Fracpete (pythonwekawrapper at gmail dot com)

import os
import sys
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.associations import Associator
import javabridge
from javabridge import JWrapper


def main(args):
    """
    Trains Apriori on the specified dataset (uses vote UCI dataset if no dataset specified).
    :param args: the commandline arguments
    :type args: list
    """

    # load a dataset
    if len(args) <= 1:
        data_file = helper.get_data_dir() + os.sep + "vote.arff"
    else:
        data_file = args[1]
    helper.print_info("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    # build Apriori, using last attribute as class attribute
    apriori = Associator(classname="weka.associations.Apriori", options=["-c", "-1"])
    apriori.build_associations(data)
    print(str(apriori))

    # iterate association rules (low-level)
    helper.print_info("Rules (low-level)")
    # make the underlying rules list object iterable in Python
    rules = javabridge.iterate_collection(apriori.jwrapper.getAssociationRules().getRules().o)
    for i, r in enumerate(rules):
        # wrap the Java object to make its methods accessible
        rule = JWrapper(r)
        print(str(i+1) + ". " + str(rule))
        # output some details on rule
        print("   - consequence support: " + str(rule.getConsequenceSupport()))
        print("   - premise support: " + str(rule.getPremiseSupport()))
        print("   - total support: " + str(rule.getTotalSupport()))
        print("   - total transactions: " + str(rule.getTotalTransactions()))

    # iterate association rules (high-level)
    helper.print_info("Rules (high-level)")
    print("can produce rules? " + str(apriori.can_produce_rules()))
    print("rule metric names: " + str(apriori.rule_metric_names))
    rules = apriori.association_rules()
    if rules is not None:
        print("producer: " + rules.producer)
        print("# rules: " + str(len(rules)))
        for i, rule in enumerate(rules):
            print(str(i+1) + ". " + str(rule))
            # output some details on rule
            print("   - consequence support: " + str(rule.consequence_support))
            print("   - consequence: " + str(rule.consequence))
            print("   - premise support: " + str(rule.premise_support))
            print("   - premise: " + str(rule.premise))
            print("   - total support: " + str(rule.total_support))
            print("   - total transactions: " + str(rule.total_transactions))
            print("   - metric names: " + str(rule.metric_names))
            print("   - metric values: " + str(rule.metric_values))
            print("   - metric value 'Confidence': " + str(rule.metric_value('Confidence')))
            print("   - primary metric name: " + str(rule.primary_metric_name))
            print("   - primary metric value: " + str(rule.primary_metric_value))
            #print("   - equals first: " + str(rule == rules[0]))
            #print("   - greater than first: " + str(rule > rules[0]))
            #print("   - greater or equal than first: " + str(rule >= rules[0]))

if __name__ == "__main__":
    try:
        jvm.start()
        main(sys.argv)
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
