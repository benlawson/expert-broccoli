#standard libraries
import glob

#external libraries
import iptcinfo
import pandas as pd
                                                                                                                                                                          #get paths to photos
filenames = glob.glob("./BUSampleDataSet/*.jpg")

filename2keywords = dict()
                                                                                                                                                                          for filename in filenames:
    keywords = iptcinfo.IPTCInfo(filename).keywords
    filename2keywords[filename[18:]] = keywords

pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in filename2keywords.items() ])).T.to_csv("tags.csv", header=False)
