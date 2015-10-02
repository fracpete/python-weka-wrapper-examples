# python-weka-wrapper-examples

Example code for the Python wrapper for Weka (https://github.com/fracpete/python-weka-wrapper).

Some of the examples are modelled after the original Examples for Weka (https://svn.cms.waikato.ac.nz/svn/weka/trunk/wekaexamples/).

Requirements:

* Python
 * python-weka-wrapper (>= 0.2.0)
* JDK 1.6+

The Python libraries you can either install using `pip install <name>` or use
pre-built packages available for your platform.


## Forum

You can post questions, patches or enhancement requests in the following Google Group:

https://groups.google.com/forum/#!forum/python-weka-wrapper

## Library Setup in Eclipse PyDev

1. when add external library, path should stop at "site-packages", and then import will work as "import weka.core.jvm as jvm" 

	working: "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages"
	not working: "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/weka"
	
2. when add source folder as library, same as above, path stop at "src", then then imporit will work as "import wekaexamples.helper as helper"

	working: "${PROJ_DIR_NAME}/src"
	not working: "${PROJ_DIR_NAME}/src/wekaexamples"