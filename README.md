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


#### Hadoop

Install hadoop, then:

    bin/hdfs dfs -mkdir /svm
