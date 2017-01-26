# CSV must be in of the form [Key,Value]
# E.g.:
#
# img_1.jpg,0
# img_2.jpg,1
# img_3.jpg,0
# img_4.jpg,0
# ...

import csv

def csvToDict(csvIn):
    with open(csvIn, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = dict((rows[0],rows[1]) for rows in reader)
        return mydict
