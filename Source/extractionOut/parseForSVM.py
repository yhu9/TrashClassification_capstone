#!/usr/bin/python

import sys
import re

if len(sys.argv) == 3:
    filenamein = sys.argv[1]
    filenameout = sys.argv[2]

    if len(filenamein) == 0 or len(filenameout) == 0:
        print("bad files passed")
        print "expected: "
        print "file_directory_in file_directory_out"
        sys.exit()

    with open(filenamein, 'r') as fin, open(filenameout,'w') as fout:
        lines = fin.read().splitlines()
        for l in lines:
            tokens = re.findall("\d+\.\d+",l)
            fout.write("+1")
            fout.write(" ")
            for i,t in enumerate(tokens):
                fout.write(str(i + 1))
                fout.write(":")
                fout.write(str(t))
                fout.write(" ")
            fout.write("\n")
else:
    print "error with the number of arguments"
    print ("length of arguments passed was: %i" % len(sys.argv))
    print "need (fileIn, fileOut)"
