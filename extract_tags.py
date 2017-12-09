#standard libraries
import glob

#external libraries
import iptcinfo
# import pandas
from sklearn.preprocessing import MultiLabelBinarizer
import os
import joblib

'''
NOTE: iptcinfo package only works with Python 2
uncomment code for writing tags to csv file
'''
filenames = []
for folder in sorted(glob.glob("WinEarthPhotosByKeyword/*")):
    print(folder)
    f = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    filenames.extend(f)

y = []
# filename2keywords = {}
for filename in filenames:
    keywords = iptcinfo.IPTCInfo(filename).keywords
    y.append(set(keywords))
    # img_name = filename.split('/')[-1]
    # filename2keywords[img_name] = keywords

encoder = MultiLabelBinarizer()
one_hot_encoded = encoder.fit_transform(y)
joblib.dump((one_hot_encoded, y, encoder), 'labels_multiclass.joblib')
print('extrated {} tags from {} images'.format(one_hot_encoded.shape[1], len(filenames)))
# df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in filename2keywords.items() ])).T
# df.to_csv('tags.csv', header=False)


