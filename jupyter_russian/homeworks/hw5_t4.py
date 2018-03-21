from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator

class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=3, max_features=10, random_state=17):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        # в данном списке будем хранить отдельные деревья
        self.trees = []
        # тут будем хранить списки индексов признаков, на которых обучалось каждое дерево 
        self.feat_ids_by_tree = []
        
    def fit(self, X, y):
        print('!!!! fit !!!!')
        for i in range (0, self.n_estimators):
            print ('====', i, '====')
            np.random.seed(self.random_state + i)
            #numOfSamples = min(X.shape[1], self.max_features)
            numOfSamples = min(X.shape[1], self.max_features)
            #print(numOfSamples)
            self.feat_ids_by_tree.append(np.random.choice(X.shape[1], numOfSamples, replace=False))
            #print('self.feat_ids_by_tree[i]', self.feat_ids_by_tree[i])
            Xloc = X[:, self.feat_ids_by_tree[i]] #подматрица по выбранным признакам
            #print("Xloc ", Xloc)
            get_bootstrap_samples(Xloc,1000)
            #tree_grid = GridSearchCV(dt, tree_params, cv=skf, scoring='roc_auc') 
            #tree_grid.fit(X, y)
            self.trees.append(DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth,\
                                        max_features=self.max_features, class_weight='balanced'))
            #print('X.shape[1] = ', Xloc.shape[1], '\t', 'max_features = ', self.max_features)
            self.trees[i].fit(Xloc, y)
        return self
        
    
    def predict_proba(self, X):
        print('!!!! predict_proba !!!!')
        pred = []
        for i in range (0, self.n_estimators):
            #print('')
            feats = X[:, self.feat_ids_by_tree[i]] #индексы признаков, на которых обучалось дерево
          #  pred += self.trees[i].predict_proba(feats)
            if i == 0:
                pred = np.array(self.trees[i].predict_proba(feats))
            else:
                pred += np.array(self.trees[i].predict_proba(feats))
            print('====', i, '====')
            #print(pred)
        '''
        и сделать прогноз вероятностей (predict_proba уже для дерева). 
        Метод должен вернуть усреднение прогнозов по всем деревьям.'''
        pred = pred/self.n_estimators
        print('#### pred ####')
        print(pred)
        return pred

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
#новый экземпляр класса
my_forest = RandomForestClassifierCustom(max_depth=7, max_features=6)
my_forest.fit(X.values, y.values)
tmp = my_forest.predict_proba(X.values)
score = roc_auc_score(y,tmp[:, 1])
#score = cross_val_score(my_forest, X.values, y, scoring='roc_auc')
print('score = ', score, '=>', round(score, 3))

'''
!!!! fit !!!!
==== 0 ====
==== 1 ====
==== 2 ====
==== 3 ====
==== 4 ====
==== 5 ====
==== 6 ====
==== 7 ====
==== 8 ====
==== 9 ====
!!!! predict_proba !!!!
==== 0 ====
==== 1 ====
==== 2 ====
==== 3 ====
==== 4 ====
==== 5 ====
==== 6 ====
==== 7 ====
==== 8 ====
==== 9 ====
#### pred ####
[[0.84867165 0.15132835]
 [0.79938451 0.20061549]
 [0.62852843 0.37147157]
 ...
 [0.57840922 0.42159078]
 [0.36840122 0.63159878]
 [0.23811865 0.76188135]]
score =  0.8420072719454691 => 0.842
'''
