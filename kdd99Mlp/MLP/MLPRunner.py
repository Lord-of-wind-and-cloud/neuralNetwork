from mongoConnect import DB_manager
from MLPTrainer import Trainer
from MLPPredictor import Predictor

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class MLPRunner:
    trainer = Trainer()
    predictor = Predictor()

    def data_load(self):
        dataset, datatarget, T_len = self.db.MLP_fetch_data()
        return dataset, datatarget, T_len

    def train(self, dataset, datatarget, T_len):
        data_set = dataset[0:(T_len - 1)]
        data_target = datatarget[0:(T_len - 1)]

        clf = ExtraTreesClassifier()
        clf = clf.fit(data_set, data_target)
        print (clf.feature_importances_)

        model = SelectFromModel(clf, prefit=True)
        feature_set = model.transform(data_set)
        datasetPlus = self.trainer.feature_selection(data_set, feature_set)
        dataSet, data_target, test_set, test_target = self.trainer.one_hot_encoding(datasetPlus, datatarget, T_len)
        self.trainer.train(dataSet, data_target)
        training_set, training_target, test_set, test_target = self.trainer.corss_validation_filter(dataSet, data_target)
        self.predictor.predict(test_set, test_target)
if __name__ == '__main__':
    runner = MLPRunner()
    dataset, datatarget, T_len = DB_manager().MLP_fetch_data()
    runner.train(dataset, datatarget, T_len)
