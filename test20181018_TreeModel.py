from sklearn import tree
import csv
from M146.HW1.src.util import *
from sklearn import metrics
import graphviz

titanic = load_data("titanic_train.csv", header=1, predict_col=0)
Xdata = titanic.X; Xnames = titanic.Xnames
ydata = titanic.y; yname = titanic.yname
n,d = Xdata.shape

clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=100)
clf = clf.fit(Xdata,ydata)
y_pred = clf.predict(Xdata)
train_error = 1 - metrics.accuracy_score(ydata, y_pred, normalize=True)
print(train_error)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=titanic.Xnames,
                                filled=True, rounded=True, class_names=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)

# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.02

# Load data

for pairidx, pair in enumerate([[2, 4], [0, 2], [1, 3],
                                [3, 5], [3, 4], [2, 6]]):
    # We only take the two corresponding features
    X = Xdata[:, pair]
    y = ydata

    # Train
    clf = tree.DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.summer)

    plt.xlabel(Xnames[pair[0]])
    plt.ylabel(Xnames[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=yname[i],
                    cmap=plt.cm.Paired, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree on Titanic Data")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()