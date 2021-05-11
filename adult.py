import re
import numpy as np

from dataset import Dataset


class Adult(Dataset):
    def __init__(self, root='./data', train=True):
        self.path = root + '/adult'
        self.train = train
        self.head = []
        self.prediction = 'income'
        self.sensitive_attr = 'sex'
        self.continuous = []
        self.missing_info = ['workclass', 'occupation', 'native-country']

        self.prediction_map = dict()
        self.sensitive_attr_map = dict()
        self.attribute_map = dict()

        # build three maps and obtain continuous variables
        self._build_maps()
        x, y, a = self._get_data()
        transform = None

        # super(Adult, self).__init__(x, y, a, transform)
        x = x.astype(np.float32)
        super(Adult, self).__init__(x, y)

    def _build_maps(self):
        f = open(self.path + '/' + 'adult.names', 'r')
        while True:
            line = f.readline()
            if line[0] == '>':
                break

        line = line[: len(line)-2]
        result = line.split(', ')
        for idx, key in enumerate(result):
            self.prediction_map[key] = idx

        f.readline() # read out blank line

        cnt = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line[: len(line)-2]
            result = re.split(': |, ', line)

            self.head.append(result[0])
            if result[0] == self.sensitive_attr:
                self.attribute_map[result[0]] = cnt
                cnt += 1
                for i in range(1, len(result)):
                    self.sensitive_attr_map[result[i]] = i-1

            elif result[1] == 'continuous':
                self.continuous.append(result[0])
                self.attribute_map[result[0]] = cnt
                cnt += 1

            else:
                for i in range(1,len(result)):
                    self.attribute_map[result[0] + '_' + result[i]] = cnt
                    cnt += 1
                if result[0] in self.missing_info:
                    self.attribute_map[result[0] + '_' + '?'] = cnt
                    cnt += 1

        self.head.append(self.prediction)

    def _get_data(self):
        if self.train:
            f = open(self.path + '/' + 'adult.data')
        else:
            f = open(self.path + '/' + 'adult.test')
            f.readline()

        x, y, a = [], [], []
        while True:
            line = f.readline()
            if len(line) == 1: continue
            if len(line) == 0: break
            if self.train:
                line = line[:len(line)-1]
            else:
                line = line[:len(line)-2]
            result = re.split(', ', line)

            xx = [0 for i in range(len(self.attribute_map))]
            yy, aa = 0, 0
            for key, value in zip(self.head, result):
                if key == self.prediction:
                    yy = self.prediction_map[value]

                elif key in self.continuous:
                    xx[self.attribute_map[key]] = int(value)

                elif key == self.sensitive_attr:
                    aa = self.sensitive_attr_map[value]
                    xx[self.attribute_map[key]] = self.sensitive_attr_map[value]

                else:
                    xx[self.attribute_map[key+'_'+value]] = 1

            x.append(xx)
            y.append(yy)
            a.append(aa)

        x = np.array(x)
        y = np.array(y)
        a = np.array(a)

        return x, y, a
