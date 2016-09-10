import argparse
import re
from subprocess import call


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set Spark Server master')
    parser.add_argument("--sparkConfFile", default="/home/s/shalaby/spark/conf/spark-env.sh")
    parser.add_argument("--bashrcFile", default="/home/s/shalaby/.bashrc")
    parser.add_argument("--server", choices=["deka","hekto"])
    args = parser.parse_args()

    if args.server == "deka":
        args.server = "deka.cip.ifi.lmu.de"
    elif args.server == "hekto":
        args.server = "hekto.cip.ifi.lmu.de"

    ## Spark Conf File
    f = open(args.sparkConfFile,'r')
    file_data = f.read()
    f.close()
    new_data = re.sub("SPARK_MASTER_IP=[\w\.]+","SPARK_MASTER_IP={}".format(args.server), file_data)
    f = open(args.sparkConfFile,'w')
    f.write(new_data)
    f.close()


    ## Bashrc File
    f = open(args.bashrcFile, 'r')
    file_data = f.read()
    f.close()
    new_data = re.sub("PYSPARK_SUBMIT_ARGS='--master spark://[\w\.]+:7077 pyspark-shell'",
                     "PYSPARK_SUBMIT_ARGS='--master spark://{}:7077 pyspark-shell'".format(args.server),
                     file_data)
    f = open(args.bashrcFile, 'w')
    f.write(new_data)
    f.close()


    ## Run Bashrc
    command = ". " + args.bashrcFile
    call(command, shell=True, executable="/bin/bash")



