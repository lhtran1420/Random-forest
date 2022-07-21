from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import StratifiedKFold


class TreeNode:
    def __init__(self, att, val):
        self.att = att
        self.val = val
        self.left = None
        self.right = None


data = genfromtxt('hw3_cancer.csv', skip_header=1)
y = []

for i in range(0, len(data), 1):
    y.append(data[i][9])

skf = StratifiedKFold(n_splits=10)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)


def Gini(a, b):
    return 1 - (a * a + b * b)


def buildDecisionTree(train_index, attribute):
    ans = 0
    val = 10
    threshd = 0

    first_class = 0
    second_class = 0
    third_class = 0

    for i in range(0, len(train_index), 1):
        if data[train_index[i]][9] == 0:
            first_class += 1
        elif data[train_index[i]][9] == 1:
            second_class += 1

    if first_class >= second_class and first_class >= third_class:
        majority = 0
    elif second_class >= first_class and second_class >= third_class:
        majority = 1

    if len(attribute) == 0:
        node = TreeNode(-1, majority)
        return node

    dt = []
    for i in range(0, len(train_index), 1):
        dt.append(data[train_index[i]])

    if len(dt) == 1:
        x = dt[0][9]
        if x == 0:
            node = TreeNode(-1, 0)
        elif x == 1:
            node = TreeNode(-1, 1)
        return node

    for i in range(0, len(attribute), 1):
        a = sorted(dt, key=itemgetter(attribute[i]))
        for j in range(0, len(a) - 1, 1):

            nums0 = 0
            nums1 = 0

            thres = (a[j][attribute[i]] + a[j + 1][attribute[i]]) / 2
            for k in range(0, j + 1, 1):
                if a[k][9] == 0:
                    nums0 += 1
                elif a[k][9] == 1:
                    nums1 += 1

            res = 0
            res = res + ((j + 1) / len(a) * Gini(nums0 / (j + 1), nums1 / (j + 1)))
            length = len(a) - j - 1
            res = res + (length / len(a) * Gini((first_class - nums0) / length, (second_class - nums1) / length))

            if res < val:
                val = res
                ans = attribute[i]
                threshd = thres

    node = TreeNode(ans, threshd)

    arr_left = []
    arr_right = []

    for i in range(0, len(train_index), 1):
        if data[train_index[i]][ans] <= threshd:
            arr_left.append(train_index[i])
        else:
            arr_right.append(train_index[i])

    if len(arr_left) == 0 or len(arr_right) == 0:
        node = TreeNode(-1, majority)
        return node

    att = []
    for i in range(0, len(attribute), 1):
        if attribute[i] != ans:
            att.append(attribute[i])

    node.left = buildDecisionTree(arr_left, att)
    node.right = buildDecisionTree(arr_right, att)
    return node


def checkPrediction(tree, obj):
    if tree.att == -1:
        return tree.val

    if obj[tree.att] <= tree.val:
        return checkPrediction(tree.left, obj)

    return checkPrediction(tree.right, obj)


def performance(train_index, test_index, ntree):
    arr = []
    for i in range(0, len(test_index), 1):
        new = []
        for j in range(0, ntree, 1):
            new.append(0)
        arr.append(new)

    for i in range(0, ntree, 1):
        attribute = np.random.choice(9, 3, replace=False)
        sub_tree = np.random.choice(train_index, len(train_index))
        tree = buildDecisionTree(sub_tree, attribute)

        for j in range(0, len(test_index), 1):
            arr[j][i] = checkPrediction(tree, data[test_index[j]])

    true_pos0 = 0
    true_pos1 = 0

    false_pos0 = 0
    false_pos1 = 0
    pos0 = 0
    pos1 = 0

    for i in range(0, len(test_index), 1):
        cnt0 = 0
        cnt1 = 0

        for j in range(0, ntree, 1):
            if arr[i][j] == 0:
                cnt0 += 1
            elif arr[i][j] == 1:
                cnt1 += 1

        if cnt0 >= cnt1:
            grp = 0
        else:
            grp = 1

        if data[test_index[i]][9] == 0:
            pos0 += 1
            if grp == 0:
                true_pos0 += 1
            else:
                false_pos1 += 1

        elif data[test_index[i]][9] == 1:
            pos1 += 1
            if grp == 1:
                true_pos1 += 1
            else:
                false_pos0 += 1

    accura = (true_pos1 + true_pos0) / len(test_index)
    precision = (true_pos1 / (true_pos1 + false_pos1) + true_pos0 / (true_pos0 + false_pos0)) / 2
    recall = (true_pos1 / pos1 + true_pos0 / pos0) / 2

    F1 = 2 * (precision * recall) / (precision + recall)

    return accura, precision, recall, F1


accura1, pre1, rec1, F1 = 0.0, 0.0, 0.0, 0.0
accura5, pre5, rec5, F5 = 0.0, 0.0, 0.0, 0.0
accura10, pre10, rec10, F10 = 0.0, 0.0, 0.0, 0.0
accura20, pre20, rec20, F20 = 0.0, 0.0, 0.0, 0.0
accura30, pre30, rec30, F30 = 0, 0, 0, 0
accura40, pre40, rec40, F40 = 0, 0, 0, 0
accura50, pre50, rec50, F50 = 0, 0, 0, 0

for train_index, test_index in skf.split(data, y):
    accuracy, precision, recall, F1_score = performance(train_index, test_index, 1)
    accura1 = accura1 + accuracy
    pre1 = pre1 + precision
    rec1 = rec1 + recall
    F1 = F1 + F1_score

    accuracy5, precision5, recall5, F15 = performance(train_index, test_index, 5)
    accura5 = accura5 + accuracy5
    pre5 = pre5 + precision5
    rec5 = rec5 + recall5
    F5 = F5 + F15

    accuracy10, precision10, recall10, F110 = performance(train_index, test_index, 10)
    accura10 = accura10 + accuracy10
    pre10 = pre10 + precision10
    rec10 = rec10 + recall10

    F10 = F10 + F110

    accuracy20, precision20, recall20, F120 = performance(train_index, test_index, 20)
    accura20 = accura20 + accuracy20
    pre20 = pre20 + precision20
    rec20 = rec20 + recall20
    F20 = F20 + F120

    accuracy30, precision30, recall30, F130 = performance(train_index, test_index, 30)
    accura30 = accura30 + accuracy30
    pre30 = pre30 + precision30
    rec30 = rec30 + recall30
    F30 = F30 + F130

    accuracy40, precision40, recall40, F140 = performance(train_index, test_index, 40)
    accura40 = accura40 + accuracy40
    pre40 = pre40 + precision40
    rec40 = rec40 + recall40
    F40 = F40 + F140

    accuracy50, precision50, recall50, F150 = performance(train_index, test_index, 50)
    accura50 = accura50 + accuracy50
    pre50 = pre50 + precision50
    rec50 = rec50 + recall50
    F50 = F50 + F150

print("1")
print(str(accura1 / 10), " ", str(pre1 / 10), " ", str(rec1 / 10), " ", str(F1 / 10))

print("5")
print(str(accura5 / 10), " ", str(pre5 / 10), " ", str(rec5 / 10), " ", str(F5 / 10))

print("10")
print(str(accura10 / 10), " ", str(pre10 / 10), " ", str(rec10 / 10), " ", str(F10 / 10))

print("20")
print(str(accura20 / 10), " ", str(pre20 / 10), " ", str(rec20 / 10), " ", str(F20 / 10))

print("30")
print(str(accura30 / 10), " ", str(pre30 / 10), " ", str(rec30 / 10), " ", str(F30 / 10))

print("40")
print(str(accura40 / 10), " ", str(pre40 / 10), " ", str(rec40 / 10), " ", str(F40 / 10))

print("50")
print(str(accura50 / 10), " ", str(pre50 / 10), " ", str(rec50 / 10), " ", str(F50 / 10))

x = [1, 5, 10, 20, 30, 40, 50]
y = [accura1 / 10, accura5 / 10, accura10 / 10, accura20 / 10, accura30 / 10, accura40 / 10, accura50 / 10]
plt.title("accuracy")
plt.plot(x, y)
plt.show()

x = [1, 5, 10, 20, 30, 40, 50]
y = [pre1 / 10, pre5 / 10, pre10 / 10, pre20 / 10, pre30 / 10, pre40 / 10, pre50 / 10]
plt.title("precision")
plt.plot(x, y)
plt.show()

x = [1, 5, 10, 20, 30, 40, 50]
y = [rec1 / 10, rec5 / 10, rec10 / 10, rec20 / 10, rec30 / 10, rec40 / 10, rec50 / 10]
plt.title("recall")
plt.plot(x, y)
plt.show()

x = [1, 5, 10, 20, 30, 40, 50]
y = [F1 / 10, F5 / 10, F10 / 10, F20 / 10, F30 / 10, F40 / 10, F50 / 10]
plt.title("F1_score")
plt.plot(x, y)
plt.show()
