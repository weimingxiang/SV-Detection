from sklearn import svm
from utilities import IdentifyDataset


def image_svm(positive_img, negative_img):
    train_proportion = 0.8
    x = torch.cat((positive_img, negative_img), 0)
    dataset_size = len(positive_img)
    y = torch.zeros(dataset_size * 2)
    y[: dataset_size] = 1
    indices = list(range(len(y)))
    split = int(np.floor(train_proportion * len(y)))
    random.seed(10)
    random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    trainx, trainy= x[train_indices], y[train_indices]
    testx, testy= x[test_indices], y[test_indices]


    clf = svm.SVC(kernel='rbf',gamma=0.001,C=100)
    # clf.fit(X_train, y_train)#训练
    # predictions0 = clf.predict(X_test)
