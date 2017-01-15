import argparse
import re
from subprocess import call


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set Spark Server master')
    parser.add_argument("--bashrcFile", default="/home/stud/shalaby/.bashrc")
    parser.add_argument("--server", choices=["vivara","yell"])
    args = parser.parse_args()

    location = ""
    if args.server == "vivara":
        location = "/home/local/shalaby"
    elif args.server == "yell":
        location = "/mnt/data2/shalaby"

    ## Bashrc File
    f = open(args.bashrcFile, 'r')
    file_data = f.read()
    f.close()
    new_data = re.sub("export IPYTHONDIR=.*",
                      "export IPYTHONDIR={}".format(location + "/ipythondir"), file_data)
    new_data = re.sub("export JUPYTER_DATA_DIR=.*",
                      "export JUPYTER_DATA_DIR={}".format(location + "/jupyterdir"), new_data)
    f = open(args.bashrcFile, 'w')
    f.write(new_data)
    f.close()


    ## Run Bashrc
    command = ". " + args.bashrcFile
    call(command, shell=True, executable="/bin/bash")
