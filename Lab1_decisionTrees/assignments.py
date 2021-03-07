import monkdata as m
import decision_tree as dtree
import numpy as np
import drawtree

# Assignment 1 ===============================================================

entropy_1 = dtree.entropy(m.monk1)
entropy_2 = dtree.entropy(m.monk2)
entropy_3 = dtree.entropy(m.monk3)

# Assignment 3 ===============================================================

datasets = [m.monk1, m.monk2, m.monk3]
gains = np.zeros((3,6))

for i in range(len(datasets)):
    for j in range(len(m.attributes)):
        gain = dtree.averageGain(datasets[i], m.attributes[j])
        gains[i,j] = gain

# Assignment 5(ej grå box) ====================================================

# Split m.monk1 according to A5 whcih has the highest gain on the fist level
splited_datasets = []
attribute = m.attributes[4]
for value in attribute.values:
    split = dtree.select(m.monk1, attribute, value)
    splited_datasets.append(split)

# Split on secon level
gains_2 = []  # Index for outer list indicate the values of the previous split. The index for the inner list indicate the gain for splitting that attribute. 
for i in range(len(attribute.values)): 
    current_gains = []                   
    for current_attribute in m.attributes:
        gain = dtree.averageGain(splited_datasets[i], current_attribute)
        current_gains.append(gain)
    gains_2.append(current_gains)
    
# Split according to the highest gains and compute majority class
A51 = dtree.select(m.monk1, m.attributes[4], 1)
A52 = dtree.select(m.monk1, m.attributes[4], 2)
A53 = dtree.select(m.monk1, m.attributes[4], 3)
A54 = dtree.select(m.monk1, m.attributes[4], 4)

# Majority classes
mostCommon_A51 = dtree.mostCommon(A51)
mostCommon_A52 = dtree.mostCommon(A52)
mostCommon_A53 = dtree.mostCommon(A53)
mostCommon_A54 = dtree.mostCommon(A54)

# Assignment 5(grå box) ======================================================

tree1 = dtree.buildTree(m.monk1, m.attributes)
tree2 = dtree.buildTree(m.monk2, m.attributes)
tree3 = dtree.buildTree(m.monk3, m.attributes)

# draw trees
# drawtree.drawTree(tree1)
# drawtree.drawTree(tree2)
# drawtree.drawTree(tree3)

# Errors
E1_train = dtree.check(tree1, m.monk1)
E1_test = dtree.check(tree1, m.monk1test)
E2_train = dtree.check(tree2, m.monk2)
E2_test = dtree.check(tree2, m.monk2test)
E3_train = dtree.check(tree3, m.monk3)
E3_test = dtree.check(tree3, m.monk3test)

# Assignment 7 ===============================================================

# dtree.plotTestError(100000)
# dtree.plotTestError(10000)
dtree.plotTestError(10)

