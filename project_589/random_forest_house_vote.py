import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import StratifiedKFold

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.mid = None
        self.right = None


data = genfromtxt('hw3_house_votes_84.csv', delimiter=',', skip_header=1)

y = []

for i in range(0, 435, 1):
    y.append(data[i][16])

skf = StratifiedKFold(n_splits=10)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)


def In(a, b):
    res = 0
    if a != 0:
        res = res - a * math.log2(a)
    if b != 0:
        res = res - b * math.log2(b)
    return res


def buildDecisionTree(train_index, depth):
    if depth > 4:
        choice_0 = 0
        choice_1 = 0

        for i in range(0, len(train_index), 1):
            if data[train_index[i]][16] == 0:
                choice_0 += 1
            else:
                choice_1 += 1

        if choice_0 > choice_1:
            node = TreeNode(1000)
            node.left = None
            node.right = None
        else:
            node = TreeNode(2000)
            node.left = None
            node.right = None
        return node

    ans = 0
    val = 10

    attribute = np.random.choice(16, 4, replace=False)
    for i in range(0, len(attribute), 1):
        first_choice = 0
        second_choice = 0
        third_choice = 0
        nums00 = 0
        nums01 = 0
        nums02 = 0

        for j in range(0, len(train_index), 1):
            x = train_index[j]
            y = attribute[i]
            if data[x][y] == 0:
                first_choice += 1
                if data[train_index[j]][16] == 0:
                    nums00 += 1

            elif data[train_index[j]][attribute[i]] == 1:
                second_choice += 1
                if data[train_index[j]][16] == 0:
                    nums01 += 1

            else:
                third_choice += 1
                if data[train_index[j]][16] == 0:
                    nums02 += 1

        res = 0
        if first_choice > 0:
            res += first_choice / len(train_index) * In(nums00 / first_choice, (first_choice - nums00) / first_choice)

        if second_choice > 0:
            res += second_choice / len(train_index) * In(nums01 / second_choice,
                                                         (second_choice - nums01) / second_choice)

        if third_choice > 0:
            res += third_choice / len(train_index) * In(nums02 / third_choice, (third_choice - nums02) / third_choice)

        if res < val:
            val = res
            ans = attribute[i]

    node = TreeNode(ans)
    arr_left = []
    arr_mid = []
    arr_right = []

    for i in range(0, len(train_index), 1):
        if data[train_index[i]][ans] == 0:
            arr_left.append(train_index[i])
        elif data[train_index[i]][ans] == 1:
            arr_mid.append(train_index[i])
        elif data[train_index[i]][ans] == 2:
            arr_right.append(train_index[i])

    if len(arr_left) > 0:
        node.left = buildDecisionTree(arr_left, depth + 1)
    if len(arr_mid) > 0:
        node.mid = buildDecisionTree(arr_mid, depth + 1)
    if len(arr_right) > 0:
        node.right = buildDecisionTree(arr_right, depth + 1)

    return node


def checkPrediction(tree, obj):
    if tree.val == 1000:
        return 0

    if tree.val == 2000:
        return 1

    value = tree.val
    if obj[value] == 0 and tree.left is not None:
        return checkPrediction(tree.left, obj)
    if obj[value] == 1 and tree.mid is not None:
        return checkPrediction(tree.mid, obj)
    if obj[value] == 2 and tree.right is not None:
        return checkPrediction(tree.right, obj)
    return 0


def performance(train_index, test_index, ntree):
    arr = []
    for i in range(0, len(test_index), 1):
        new = []
        for j in range(0, ntree, 1):
            new.append(0)
        arr.append(new)

    for i in range(0, ntree, 1):
        attribute = np.random.choice(16, 4, replace=False)
        sub_tree = np.random.choice(train_index, len(train_index))

        # print(sub_tree)
        # print(attribute)
        tree = buildDecisionTree(sub_tree, 1)

        for j in range(0, len(test_index), 1):
            arr[j][i] = checkPrediction(tree, data[test_index[j]])

    true_pos = 0
    true_neg = 0
    pos = 0
    neg = 0

    for i in range(0, len(test_index), 1):
        if (data[test_index[i]][16] == 0):
            neg += 1
        elif (data[test_index[i]][16] == 1):
            pos += 1

    for i in range(0, len(test_index), 1):
        cnt0 = 0
        cnt1 = 0

        for j in range(0, ntree, 1):
            if (arr[i][j] == 0):
                cnt0 += 1
            else:
                cnt1 += 1
        if (cnt0 >= cnt1):
            grp = 0
        else:
            grp = 1

        if (data[test_index[i]][16] == 0):
            if (grp == 0):
                true_neg += 1

        if (data[test_index[i]][16] == 1):
            if (grp == 1):
                true_pos += 1

    accura = (true_pos + true_neg) / len(test_index)
    precision = true_pos / pos
    recall = true_pos / (true_pos + (neg - true_neg))
    F1 = 2 * (precision * recall) / (precision + recall)

    return accura, precision, recall, F1


accura1, pre1, rec1, F1 = 0, 0, 0, 0
accura5, pre5, rec5, F5 = 0, 0, 0, 0
accura10, pre10, rec10, F10 = 0, 0, 0, 0
accura20, pre20, rec20, F20 = 0, 0, 0, 0
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

# print("\n")
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
