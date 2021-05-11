import pandas as pd
import numpy as np
import pickle

from dataset import Dataset
# from utils import Dataset


class Compas(Dataset):
    def __init__(self, root='./data', train=True):
        self.root = root + '/compas'
        self.train = train

        if self.train:
            self.f_list_path = root + '/compas/indices/compas-train-list.pkl'
        else:
            self.f_list_path = root + '/compas/indices/compas-test-list.pkl'

        with open(self.f_list_path, 'rb') as f:
            self.f_list = pickle.load(f)

        ''' the selection of features follows
        https://github.com/IBM/AIF360/blob/master/aif360/datasets/compas_dataset.py
        '''
        # self.selected_features = ['age', 'c_charge_degree', 'race', 'age_cat',
        #     'score_text', 'sex', 'priors_count', 'days_b_screening_arrest',
        #     'decile_score', 'two_year_recid']
        self.selected_features = ['sex', 'age', 'age_cat', 'race',
             'juv_fel_count', 'juv_misd_count', 'juv_other_count',
             'priors_count', 'c_charge_degree', 'c_charge_desc',
             'two_year_recid']

        self.sensitive_attr = 'race'
        self.prediction = 'two_year_recid'
        self.categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']

        x, y, a = self._get_data()
        transform = None

        x = x[self.f_list]
        y = y[self.f_list]
        a = a[self.f_list]

        # train_sz = int(len(x) * 0.7)
        # total_list = np.random.permutation( len(x) )
        # train_list = total_list[:train_sz]
        # test_list = total_list[train_sz:]
        # with open(path + '/index/compas-train-list.pkl', 'wb') as f:
        #     pickle.dump(train_list, f)
        # with open(path + '/index/compas-test-list.pkl', 'wb') as f:
        #     pickle.dump(test_list, f)
        # exit()

        # super(Compas, self).__init__(x, y, a, transform)
        x = x.astype(np.float32)
        super(Compas, self).__init__(x, y) 

    def _get_data(self):
        ''' the following preprocessing is almost the same as in
        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        '''
        df = pd.read_csv('{}/compas-scores-two-years.csv'.format(self.root))
        df = df[self.selected_features]
        df = df.dropna()
        # df = df[ df['days_b_screening_arrest'] <= 30 ]
        # df = df[ df['days_b_screening_arrest'] >= -30 ]
        df = df[ df['c_charge_degree'] != 'O' ]
        df = df[ df['c_charge_degree'] != -1 ]
        # print(df)
        # print(len(df))

        df['race'] = df['race'].apply(lambda x: 1 if x == 'African-American' else 0)
        df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
        df = pd.get_dummies(df, columns=self.categorical_features, prefix_sep='=')

        y = np.array(df[self.prediction])
        a = np.array(df[self.sensitive_attr])

        df = df.drop(self.prediction, axis=1)
        x = np.array(df)

        return x, y, a
