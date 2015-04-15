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

# filters.py
# Copyright (C) 2014-2015 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.core.stemmers import Stemmer
from weka.core.stopwords import Stopwords
from weka.core.tokenizers import Tokenizer
from weka.filters import Filter, MultiFilter, StringToWordVector


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(iris)

    # remove class attribute
    helper.print_info("Removing class attribute")
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
    remove.inputformat(data)
    filtered = remove.filter(data)

    # use MultiFilter
    helper.print_info("Use MultiFilter")
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    std = Filter(classname="weka.filters.unsupervised.attribute.Standardize")
    multi = MultiFilter()
    multi.filters = [remove, std]
    multi.inputformat(data)
    filtered_multi = multi.filter(data)

    # output datasets
    helper.print_title("Input")
    print(data)
    helper.print_title("Output")
    print(filtered)
    helper.print_title("Output (MultiFilter)")
    print(filtered_multi)

    # load text dataset
    text = helper.get_data_dir() + os.sep + "reutersTop10Randomized_1perc_shortened.arff"
    helper.print_info("Loading dataset: " + text)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(text)
    data.class_is_last()

    # apply StringToWordVector
    stemmer = Stemmer(classname="weka.core.stemmers.IteratedLovinsStemmer")
    stopwords = Stopwords(classname="weka.core.stopwords.Rainbow")
    tokenizer = Tokenizer(classname="weka.core.tokenizers.WordTokenizer")
    s2wv = StringToWordVector(options=["-W", "10", "-L", "-C"])
    s2wv.stemmer = stemmer
    s2wv.stopwords = stopwords
    s2wv.tokenizer = tokenizer
    s2wv.inputformat(data)
    filtered = s2wv.filter(data)

    helper.print_title("Input (StringToWordVector)")
    print(data)
    helper.print_title("Output (StringToWordVector)")
    print(filtered)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
