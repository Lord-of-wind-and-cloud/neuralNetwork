#from sklearn.externals import joblib
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Predictor:
    def predict(self, test_set, test_target):
        clf = joblib.load('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/mlp/MLP.pkl')
        trained_target = clf.predict(test_set)
        # confusion_matrix's diagonal is the accurate number
        print (confusion_matrix(test_target, trained_target, labels=[0, 1, 2, 3, 4]))
        # precision    recall  f1-score   support
        print (classification_report(test_target, trained_target))





