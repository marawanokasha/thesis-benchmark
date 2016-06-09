## Setup

#### First for Scipy:

	sudo apt-get install build-essential gfortran libatlas-base-dev
	
#### For the General Requirements:

Then:

	pip install virtualenv virtualenvwrapper
	vim ~/.bashrc

Then add those two lines there:

	export WORKON_HOME=$HOME/.virtualenv
	source /usr/local/bin/virtualenvwrapper.sh

Then:

	source ~/.bashrc
	mkvirtualenv thesis-env
	workon thesis-env
	pip install -r requirements.txt
	
#### For NLTK:

Do 

	workon thesis-env
	
Then go into a python console, then:

	import nltk
	nltk.download("wordnet")
	nltk.download("stopwords")
	
#### For Spark

Open ~/.bashrc and add
	
	export SPARK_HOME={wherever you put your spark}
	
then

    cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh
    echo "export PYSPARK_PYTHON=/home/{username_here}/.virtualenv/thesis-env/bin/python" >> $SPARK_HOME/conf/spark-env.sh

You can setup your spark configuration by editing conf/spark-env.sh for example by adding the master ip and path to python to use

    SPARK_MASTER_IP=192.168.0.103
    SPARK_WORKER_CORES=3
    SPARK_WORKER_INSTANCES=1
    export PYSPARK_PYTHON=/home/marawan/.virtualenv/thesis-env/bin/python
    export PYSPARK_DRIVER_PYTHON=/home/marawan/.virtualenv/thesis-env/bin/python

and conf/spark-defaults.conf to configure the worker memory:

    spark.executor.memory   2g

you can then run the spark master using:

    sbin/start-master.sh
    
and then the slaves can be run using:

    sbin/start-slave.sh spark://192.168.0.103:7077
    
#### For Ipython

    ipython profile create pyspark
    vim ~/.ipython/profile_pyspark/startup/00-pyspark-setup.py
    
And add the following:
    
    import os
    import sys
    
    spark_home = os.environ.get('SPARK_HOME', None)
    sys.path.insert(0, spark_home + "/python")
    sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.9-src.zip'))
    
    filename = os.path.join(spark_home, 'python/pyspark/shell.py')
    exec(compile(open(filename, "rb").read(), filename, 'exec'))
    
    spark_release_file = spark_home + "/RELEASE"
    
    if os.path.exists(spark_release_file) and "Spark 1.6" in open(spark_release_file).read():
        pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
        if not "pyspark-shell" in pyspark_submit_args:
            pyspark_submit_args += " pyspark-shell"
            os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

You may need to change the py4j-0.x and Spark 1.x depending on your version

then in ~/.bashrc add:

    export PYSPARK_SUBMIT_ARGS='--master spark://131.159.195.118:7077 pyspark-shell'
    
*pyspark-shell* must be there for the correct functioning of pyspark within ipython

then you can just do 

    ipython --profile=pyspark
    
to work in the console

To work in jupyter notebook, you need to create a file: 

    mkdir -p .ipython/kernels/pyspark/
    vim .ipython/kernels/pyspark/kernel.json
    
and add the following:

    {
     "display_name": "pySpark (Spark 1.6.1)",
     "language": "python",
     "argv": [
      "/home/marawan/.virtualenv/thesis-env/bin/python",
      "-m",
      "IPython.kernel",
      "--profile=pyspark",
      "-f",
      "{connection_file}"
     ]
    }

then you can just run 

    jupter notebook
    
and when creating a new notebook, choose the one with the type pySpark.

#### Hadoop

Install hadoop, then:

    bin/hdfs dfs -mkdir /svm
