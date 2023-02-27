import numpy as np


def entropy(li: np.array) -> float:

    # if all elements are same the entropy will be zero
    if len(set(li)) == 1:
        return 0
    
    # counting the categorical counts within the array
    cat_count = {j: list(li).count(j) for j in set(li)}
    
    try:
        # Entropy = -∑(pi*log2pi) where pi=count(xi)/count(x)
        return sum([(-(cat_count[k]/len(li))*(np.log2(cat_count[k]/len(li)))) for k in list(cat_count.keys())])
    except:
        raise Exception("Cannot calculate entropy for this array")



def gain(feature_x: np.array, outcome_y: np.array) -> float:

    # gain = entropy(outcome_y) - ∑(count(feature_xi)/count(feature_x) * entropy(feature_yi))
    # where i is a category in feature x
    
    # calulating entropy of outcome
    s = entropy(outcome_y)

    # category counts of feature x
    x_cat_count = {j: list(feature_x).count(j) for j in set(feature_x)}
    
    # Iterating over category in feature column and subtracting the ratio of category with the entropy
    for c in range(len(list(x_cat_count.keys()))):
        
        indices = []
        for i in range(len(feature_x)):
            if feature_x[i] == c:
                indices.append(i)
        
        fy = outcome_y[indices]
        
        sub_entropy = (len(indices)/len(feature_x))*entropy(fy)
        
        s -= sub_entropy

    return s