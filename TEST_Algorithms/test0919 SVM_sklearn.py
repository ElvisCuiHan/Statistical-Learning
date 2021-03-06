import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from TEST_Algorithms.loadDataSet import ringDataSet


def plot_decision_regions(X, y, classifier, resolution=0.02):

    colors = ('lightgreen', 'cyan', 'gray', 'r', 'b')
    markers = ('s', 'x', 'o', '^', 'v')


    x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    XX, YY = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([XX.ravel(), YY.ravel()]).T)
    Z = Z.reshape(XX.shape)
    plt.contourf(XX, YY, Z, alpha=.4)
    plt.xlim((XX.min(), XX.max()))
    plt.ylim((YY.min(), YY.max()))

    # plot class samples
    for idx, l in enumerate(np.unique(y)):
        plt.scatter(X[y == l, 0], X[y == l, 1], alpha=0.8, marker=markers[idx], label=l)

    plt.savefig('image/result.png')

dataMat,labelMat = ringDataSet()
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataMat, labelMat, test_size=0.25, random_state=5)

params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
plot_decision_regions(X_test, y_test, classifier)
target_names = ['Class-' + str(int(i)) for i in set(labelMat)]
print((classification_report(y_test, classifier.predict(X_test))))