#An example decision tree making with input data 
import numpy as np
import matplotlib.pyplot as plt
#DEFINE THE TRAINING DATA SETS
x_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
print(x_train.shape)
print(y_train.shape)
#define compute entropy 
def compute_entropy(y):
    entropy =0
    p1=0
    m = len(y)
    if m != 0:
        for i in range (m):
            if y[i] == 1:
                p1=p1+1
        p1 = p1/m
    if p1 !=0 and p1!=1:
        entropy = -p1*np.log2(p1)-(1-p1)*np.log2(1-p1)
    else:
        entropy =0
    return entropy
print(compute_entropy(y_train))
#define split dataset. Dependent upon the feature's value (0, 1) split the dataset into two smaller datasets in order to reduce the entropy.
def split_datasets_innode(x,node_indices,feature):
    left_index=[]
    right_index=[]
    for i in range(len(node_indices)):
        #print(node_indices[i])
        if x[node_indices[i]][feature] == 1:
            left_index.append(node_indices[i])
        else:
            right_index.append(node_indices[i])
    return left_index, right_index
#feature=0
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#print(split_datasets_innode(x_train, root_indices, feature=0))


#we now need to put together all along but before that we need to calculate information gain for the features to select for the nodes
def compute_information_gain(x, y, node_indices, feature):
    information_gain =0
    #x_node = x_train[node_indices]
    #y_node = y_train[node_indices]
    left_index, right_index=split_datasets_innode(x, node_indices, feature)
    print(left_index, right_index)
    x_node = np.zeros(len(node_indices))
    y_node = np.zeros(len(node_indices))
    x_left = np.zeros(len(left_index))
    y_left = np.zeros(len(left_index))
    x_right = np.zeros(len(right_index))
    y_right = np.zeros(len(right_index))
    count_element =-1
    for i in range(len(node_indices)):
        count_element =count_element +1
        x_node[count_element]=x_train[node_indices[i]][feature]
        y_node[count_element]=y_train[node_indices[i]]
    #print(x_node, y_node)
    count_element =-1
    for i in range(len(left_index)):
        count_element =count_element +1
        x_left[count_element]=x_train[left_index[i]][feature]
        y_left[count_element]=y_train[left_index[i]]
    
    count_element =-1
    for i in range(len(right_index)):
        count_element =count_element +1
        x_right[count_element]=x_train[right_index[i]][feature]
        y_right[count_element]=y_train[right_index[i]]
    print(x_left.shape, y_left.shape)
    
    w_left = len(left_index)/len(x_node)
    w_right = len(right_index)/len(x_node)
    print(w_left, w_right)
    node_entropy = compute_entropy(y_node)  
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    information_gain =node_entropy-w_left*left_entropy- w_right*right_entropy

    return information_gain
    

print(compute_information_gain(x_train, y_train, root_indices, feature=0))
print(compute_information_gain(x_train, y_train, root_indices, feature=1))
print(compute_information_gain(x_train, y_train, root_indices, feature=2))




