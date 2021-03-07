import math
import random
import numpy as np
import monkdata as m
import matplotlib.pyplot as plt


def entropy(dataset):
    "Calculate the entropy of a dataset"
    n = len(dataset)
    nPos = len([x for x in dataset if x.positive])
    nNeg = n - nPos
    if nPos == 0 or nNeg == 0:
        return 0.0
    return -float(nPos)/n * log2(float(nPos)/n) + \
        -float(nNeg)/n * log2(float(nNeg)/n)


def averageGain(dataset, attribute):
    "Calculate the expected information gain when an attribute becomes known"
    weighted = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        weighted += entropy(subset) * len(subset)
    return entropy(dataset) - weighted/len(dataset)


def log2(x):
    "Logarithm, base 2"
    return math.log(x, 2)


def select(dataset, attribute, value):
    "Return subset of data samples where the attribute has the given value"
    return [x for x in dataset if x.attribute[attribute] == value]


def bestAttribute(dataset, attributes):
    "Attribute with highest expected information gain"
    gains = [(averageGain(dataset, a), a) for a in attributes]
    return max(gains, key=lambda x: x[0])[1]


def allPositive(dataset):
    "Check if all samples are positive"
    return all([x.positive for x in dataset])


def allNegative(dataset):
    "Check if all samples are negative"
    return not any([x.positive for x in dataset])


def mostCommon(dataset):
    "Majority class of the dataset"
    pCount = len([x for x in dataset if x.positive])
    nCount = len([x for x in dataset if not x.positive])
    return pCount > nCount


class TreeNode:
    "Decision tree representation"

    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        "Produce readable (string) representation of the tree"
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'


class TreeLeaf:
    "Decision tree representation for leaf nodes"

    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        "Produce readable (string) representation of this leaf"
        if self.cvalue:
            return '+'
        return '-'


def buildTree(dataset, attributes, maxdepth=1000000):
    "Recursively build a decision tree"

    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        if allPositive(dataset):
            return TreeLeaf(True)
        if allNegative(dataset):
            return TreeLeaf(False)
        return buildTree(dataset, attributes, maxdepth-1)

    default = mostCommon(dataset)
    if maxdepth < 1:
        return TreeLeaf(default)
    a = bestAttribute(dataset, attributes)
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft))
                for v in a.values]
    return TreeNode(a, dict(branches), default)


def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[sample.attribute[tree.attribute]], sample)


def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)


def allPruned(tree):
    "Return a list if trees, each with one node replaced by the corresponding default class"
    if isinstance(tree, TreeLeaf):
        return ()
    alternatives = (TreeLeaf(tree.default),)
    for v in tree.branches:
        for r in allPruned(tree.branches[v]):
            b = tree.branches.copy()
            b[v] = r
            alternatives += (TreeNode(tree.attribute, b, tree.default),)
    return alternatives


def partition(data, fraction):
    """Splits training data into actual training data and into validation data.
    Fraction indicates the fraction of the datasets the becomes training data 
    and (1 - fraction) indicates the fraction of the datasets the becomes 
    validation data. data is the original training dataset."""
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]
    

def prunedTree(tree, dataset):
    """Returns the best three after pruning. The best tree is defines as the 
    one having the smallest validation error. Dataset is the validation data."""
    prunedTrees = allPruned(tree)             # List of all possible trees  
    bestTree = tree                                 # The tree with the smallest validation error
    E_validation = check(bestTree, dataset)   # Validation error
    # Iterate over all possible trees
    for currentTree in prunedTrees[1:]:
        E_validation_current = check(currentTree, dataset)
        if E_validation_current <= E_validation:
            bestTree = currentTree
    return bestTree


def prunedTreeError(N, fraction, datasetTrain, datasetTest):
    """Returns the mean test error and standard deviation of the best tree 
    after pruning has occured. datasetTrain is the originial training data that 
    is to be partitioned. datasetTest is the testing data. N is the number of 
    times the model is trained with the paritioned dataset."""
    testErrors = []
    # Compute the test error multiple times due to the randomness in the split of the data in partition
    for i in range(N):
        trainData, validationData = partition(datasetTrain, fraction)   # Split the data
        tree = buildTree(trainData, m.attributes)                 # Compute the desicion tree
        pruned_tree = prunedTree(tree, validationData)                  # the best tree after pruning has occured
        testError = check(pruned_tree, datasetTest)
        testErrors.append(testError)
    
    return np.mean(testErrors), np.std(testErrors)   


def prunedTreeErrorVersusFraction(N, datasetTrain, datasetTest):
    """Return lists of the test error and standard deviation for different 
    fractions used in partion."""
    fractions = np.arange(0.3, 0.9, 0.1)
    testErrors = []
    testStdev = []
    # Compute the test error for all fractions and store in list
    for fraction in fractions:
        error, stdev = prunedTreeError(N, fraction, datasetTrain, datasetTest)
        testErrors.append(error)
        testStdev.append(stdev)      
    return testErrors, testStdev


def plotTestError(N):
    """Plots the test error for the pruned tree as function of the fractoin 
    used in partion for the datasets monk1 and monk3."""
    monk1Error, monk1Stdev = prunedTreeErrorVersusFraction(N, m.monk1, m.monk1test)
    monk3Error, monk3Stdev = prunedTreeErrorVersusFraction(N, m.monk3, m.monk3test)
    fractions = np.arange(0.3, 0.9, 0.1)
    
    plt.figure()
    plt.title('Test error for monk1 and monk3. The error bars show\n the standard deviation.')
    plt.ylabel('Test error')
    plt.xlabel('Fraction')
    plt.errorbar(fractions, monk1Error, yerr=monk1Stdev, marker='o')
    plt.errorbar(fractions, monk3Error, yerr=monk3Stdev, marker='o')
    plt.legend(['monk1', 'monk3'])