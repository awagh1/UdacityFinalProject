#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi','salary','to_messages', 'total_payments', 'bonus', 'total_stock_value', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
import matplotlib.pyplot

data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
#data = featureFormat(data_dict, features)


### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
#print outliers_final

##Remove these 4 outliers from main disctionary
data_dict.pop(outliers_final[0][0], 0)
data_dict.pop(outliers_final[1][0], 0)
data_dict.pop(outliers_final[2][0], 0)
data_dict.pop(outliers_final[3][0], 0)

data = featureFormat(data_dict, features_list)

import matplotlib.pyplot as plt
### plot features
for point in data:
    from_poi_to_this_person = point[6]
    from_this_person_to_poi = point[9]
    plt.scatter( from_poi_to_this_person, from_this_person_to_poi )

plt.xlabel("from_poi_to_this_person")
plt.ylabel("from_this_person_to_poi")
plt.show()



### Task 3: Create new feature(s)
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

#Update/Ovewrite features List
features_list = ['poi', 'fraction_from_poi_email', 'fraction_to_poi_email', 'shared_receipt_with_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

###again remove outliners
outliers = []
for key in data_dict:
    val = data_dict[key]['fraction_from_poi_email']
    outliers.append((key, float(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:1])
#print outliers_final

my_dataset.pop(outliers_final[0][0],0)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.show()

labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=5)

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = accuracy_score(labels_test ,pred)
print 'accuracy =' , acc
# out of all positives (true + false)
print 'precision = ', precision_score(labels_test,pred)
# out of true positives and false negatives
print 'recall = ', recall_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
