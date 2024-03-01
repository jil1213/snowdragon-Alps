#script to get all classified markers out of the ini files and combine them 
# in one ini file to check the missing markers mf and ppgp
import os
from configparser import ConfigParser

def combine_ini_files(directory, output_file):
    config = ConfigParser()

    for filename in os.listdir(directory):
        if filename.endswith(".ini"):
            filepath = os.path.join(directory, filename)
            config.read(filepath)

    with open(output_file, "w") as f:
        config.write(f)

# Usage example
directory = "C:/Users/jille/Documents/LWD/snowdragon-Alps/data/smp_pnt_files"
output_file = "combined.ini"
combine_ini_files(directory, output_file)
