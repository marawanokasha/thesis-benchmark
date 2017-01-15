## Setup

#### First for Scipy:

	sudo apt-get install build-essential gfortran libatlas-base-dev libfreetype6-dev
	
#### For the General Requirements:

Then:

	pip install virtualenv virtualenvwrapper
	vim ~/.bashrc

Then add those two lines there:

	export WORKON_HOME=$HOME/.virtualenv
	source /usr/local/bin/virtualenvwrapper.sh

Or if you get Permission denied errors:

	pip install --user virtualenv virtualenvwrapper

Then add those two lines there:

	export WORKON_HOME=$HOME/.virtualenv
	source /home/stud/shalaby/.local/bin/virtualenvwrapper.sh
	# for being able to run jupyter from the command line
	export PATH=$PATH:/home/stud/shalaby/.local/bin

Then:

	source ~/.bashrc
	mkvirtualenv thesis-env
	workon thesis-env
	pip install -r requirements.txt

Or if you installed the virtualenv locally:

	pip install --user -r requirements.txt
	

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
	
	# bash function wrapping python script to set python server
    set_spark_server(){
            python ~/thesis/thesis-benchmark/src/thesis/set_spark_server.py --server $1
    }

	
then

    cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh
    echo "export PYSPARK_PYTHON=/home/{username_here}/.virtualenv/thesis-env/bin/python" >> $SPARK_HOME/conf/spark-env.sh

You can setup your spark configuration by editing `conf/spark-env.sh` for example by adding the master ip and path to python to use

    SPARK_MASTER_IP=deka.cip.ifi.lmu.de
    # the default is to use all cores
    SPARK_WORKER_CORES=20
    # this is the default anyway
    SPARK_WORKER_INSTANCES=1
    export PYSPARK_PYTHON=/home/marawan/.virtualenv/thesis-env/bin/python
    export PYSPARK_DRIVER_PYTHON=/home/marawan/.virtualenv/thesis-env/bin/python

and `conf/spark-defaults.conf` to configure the worker memory and temp directory (important if /tmp has disc restrictions, otherwise jobs fail randomly with an obscure java server error):

    spark.executor.memory   20g
    spark.driver.memory 8g
    # important when calculating chi due to the large amount of data sent to the driver
    spark.driver.maxResultSize 4g
    # Enabling applications to share resources
    spark.dynamicAllocation.enabled true
    spark.shuffle.service.enabled true
    # set path for temp files created during task execution
    spark.local.dir /big/s/shalaby/tmp
    # for accessing elasticsearch from spark
    spark.driver.extraClassPath     /home/s/shalaby/lib/elasticsearch-spark_2.10-2.3.2.jar
    spark.executor.extraClassPath   /home/s/shalaby/lib/elasticsearch-spark_2.10-2.3.2.jar


you can then run the spark master using:

    sbin/start-master.sh
    
and then the slaves can be run using:

    sbin/start-slave.sh spark://deka.cip.ifi.lmu.de:7077
   
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

    export PYSPARK_SUBMIT_ARGS='--master spark://deka.cip.ifi.lmu.de:7077 pyspark-shell'
    
*pyspark-shell* must be there for the correct functioning of pyspark within ipython

then you can just do 

    ipython --profile=pyspark
    
to work in the console

To work in jupyter notebook, you need to create a file: 

    mkdir -p ~/.ipython/kernels/pyspark/
    vim ~/.ipython/kernels/pyspark/kernel.json
    
and add the following:

    {
     "display_name": "pySpark (Spark 1.6.1)",
     "language": "python",
     "argv": [
      "/home/s/shalaby/.virtualenv/thesis-env/bin/python",
      "-m",
      "IPython.kernel",
      "--profile=pyspark",
      "-f",
      "{connection_file}"
     ]
    }

then you can just run 

    workon thesis-env
    jupyter notebook  --ip=0.0.0.0
    
to make it listen to all associated ips and not just localhost. When creating a new notebook, choose the one with the type pySpark.


#### Theano

To get theano to work in ipython, add this to `.bashrc`:

    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
    
or else you'll get `Cuda is unavailable` errors. You should also add `/usr/local/cuda-8.0/bin` to `$PATH`, but that's already taken care of on yell


#### TensorFlow

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp27-none-linux_x86_64.whl
    pip install --upgrade $TF_BINARY_URL

#### Hadoop

Download hadoop, then add to .bashrc:

    export $JAVA_HOME={your java directory}

and source it, then in **etc/hadoop/core-site.xml**:
    
    <configuration>
        <property>
            <!-- this determines the primary node -->
            <name>fs.defaultFS</name>
            <value>hdfs://deka.cip.ifi.lmu.de:8020/</value>
        </property>
        <property>
             <name>hadoop.tmp.dir</name>
             <value>/big/s/shalaby/tmp-hadoop</value>
        </property>
    </configuration>


then in **etc/hadoop/hdfs-site.xml**:
    
     <configuration>
              <property>
                      <name>dfs.replication</name>
                      <value>1</value>
              </property>
              <property>
                 <name>dfs.namenode.secondary.http-address</name>
                 <value>deka.cip.ifi.lmu.de:50090</value>
              </property>
              <property>
                      <name>dfs.namenode.name.dir</name>
                      <value>/home/s/shalaby/hdfs_content/hdfs_name_dir</value>
              </property>
              <property>
                      <name>dfs.datanode.data.dir</name>
                      <value>/home/s/shalaby/hdfs_content/data</value>
              </property>
              <property>
                      <name>dfs.namenode.checkpoint.dir</name>
                      <value>/home/s/shalaby/hdfs_content/secondary_checkpoitn_dir</value>
              </property>
      </configuration>


then in **etc/hadoop/yarn-site.xml**:

    <configuration>
             <property>
                     <name>yarn.resourcemanager.hostname</name>
                     <value>deka.cip.ifi.lmu.de</value>
             </property> 
     </configuration>


then in **etc/hadoop/slaves**:

    hekto.cip.ifi.lmu.de
    deka.cip.ifi.lmu.de

Then to setup:

    bin/hdfs namenode -format
    sbin/start-dfs.sh
    bin/hdfs dfs -mkdir /svm


### Elastic search

To connect elastic search with spark, you need to download the elasticsearch-hadoop connector jar from https://spark-packages.org/package/elastic/elasticsearch-hadoop, 
then add its path to the spark configuration in `spark-defaults.conf`

    # for accessing elasticsearch from spark
    spark.driver.extraClassPath     /home/s/shalaby/lib/elasticsearch-spark_2.10-2.3.2.jar
    spark.executor.extraClassPath   /home/s/shalaby/lib/elasticsearch-spark_2.10-2.3.2.jar

Elasticsearch itself doesn't require too much setup, you may just want to change the `network.host` property in `confing/elasticsearch.yml` so you can access the server from a url other than `localhost`:

    network.host: deka.cip.ifi.lmu.de

You can also set the Xmx and Xms flags by adding this to `.bashrc`:

    export ES_HEAP_SIZE=2g


### OPTICS

    java -jar elki-bundle-0.7.1.jar KDDCLIApplication -dbc.in /home/local/shalaby/bm25_training_sample_0.008_dense.csv -algorithm clustering.optics.OPTICSList -optics.minpts 3 -verbose -resulthandler ExportVisualizations,ResultWriter -vis.output /home/local/shalaby/optics-0.008-vis -out /home/local/shalaby/optics-0.008-text
    
### SOCKS Proxy

Setup a socks proxy using:

    ssh -D 1024 lmu-remote

Or using autossh to guarantee a continuous connection:

    autossh -ND localhost:1024 lmu-remote &

Then use FoxyProxy add-on for firefox to use this socks proxy to access GUI websites like ipython