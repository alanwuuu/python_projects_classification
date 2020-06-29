from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from matplotlib import pyplot as plt

breast_cancer_data = load_breast_cancer()

#get a sense of what the data looks like
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)

#get a sense of what the labels look like
#print(breast_cancer_data.target[0])
#print(breast_cancer_data.target_names)

#split the training and validation sets
training_data,validation_data,training_labels,validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target,test_size=0.2,random_state=100)

#ensure the lists are the same size
#print(len(training_data),len(training_labels))

#creating our classifier
classifier = KNeighborsClassifier(n_neighbors = 3)

#train our classifier
classifier.fit(training_data,training_labels)

#checking how accurate out classifier is on our validation set
#print(classifier.score(validation_data,validation_labels))

#the above sounds good with k=3, but we want to see if there is a better k
#lets do a forloop for k from 1-100

accuracy_list = []
for k in range(1,101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data,training_labels)
    accuracy= classifier.score(validation_data,validation_labels)
    accuracy_list.append(accuracy)
    
#now lets plot accuracy_list on a graph 
plt.plot(range(1,101),accuracy_list)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
