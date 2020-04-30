# Modeling
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV

nF = 20
kf = KFold(n_splits=nF, random_state=241, shuffle=True)

test_errors_l2 = []
train_errors_l2 = []
test_errors_l1 = []
train_errors_l1 = []
test_errors_GBR = []
train_errors_GBR = []
test_errors_ENet = []
test_errors_LGB = []
test_errors_stack = []
test_errors_ens = []
train_errors_ens = []

models = []

pred_all = []

ifold = 1

# 第一层基模型选择Ridge、 Lasso、ElasticNet、lightgbm四种，第二层模型选用Lasso。模型融合过程代码如下(可以先做一下网格搜索，寻找合适的算法超参)：

for train_index, test_index in kf.split(X):
    print('fold: ', ifold)
    ifold = ifold + 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # ridge
    l2Regr = Ridge(alpha=9.0, fit_intercept=True)
    l2Regr.fit(X_train, y_train)
    pred_train_l2 = l2Regr.predict(X_train)
    pred_test_l2 = l2Regr.predict(X_test)

    # lasso
    l1Regr = make_pipeline(RobustScaler(), Lasso(alpha=0.0003, random_state=1, max_iter=50000))
    l1Regr.fit(X_train, y_train)
    pred_train_l1 = l1Regr.predict(X_train)
    pred_test_l1 = l1Regr.predict(X_test)

    # GBR
    myGBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=50,
                                      loss='huber', random_state=5)

    myGBR.fit(X_train, y_train)
    pred_train_GBR = myGBR.predict(X_train)

    pred_test_GBR = myGBR.predict(X_test)

    # ENet
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
    ENet.fit(X_train, y_train)
    pred_train_ENet = ENet.predict(X_train)
    pred_test_ENet = ENet.predict(X_test)

    # LGB
    myLGB = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=600,
                              max_bin=50, bagging_fraction=0.6,
                              bagging_freq=5, feature_fraction=0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    myLGB.fit(X_train, y_train)
    pred_train_LGB = myLGB.predict(X_train)
    pred_test_LGB = myLGB.predict(X_test)

    # Stacking
    stackedset = pd.DataFrame({'A': []})
    stackedset = pd.concat([stackedset, pd.DataFrame(pred_test_l2)], axis=1)
    stackedset = pd.concat([stackedset, pd.DataFrame(pred_test_l1)], axis=1)
    stackedset = pd.concat([stackedset, pd.DataFrame(pred_test_GBR)], axis=1)
    stackedset = pd.concat([stackedset, pd.DataFrame(pred_test_ENet)], axis=1)
    stackedset = pd.concat([stackedset, pd.DataFrame(pred_test_LGB)], axis=1)
    # prod = (pred_test_l2*pred_test_l1*pred_test_GBR*pred_test_ENet*pred_test_LGB) ** (1.0/5.0)
    # stackedset = pd.concat([stackedset,pd.DataFrame(prod)],axis=1)
    Xstack = np.array(stackedset)
    Xstack = np.delete(Xstack, 0, axis=1)
    l1_staked = Lasso(alpha=0.0001, fit_intercept=True)
    l1_staked.fit(Xstack, y_test)
    pred_test_stack = l1_staked.predict(Xstack)
    models.append([l2Regr, l1Regr, myGBR, ENet, myLGB, l1_staked])

    # 模型预测
    X_score = np.array(df_score)
    X_score = np.delete(X_score, 0, 1)
    M = X_score.shape[0]
    scores_fin = 1 + np.zeros(M)
    for m in models:
        ger = m[0]
        las = m[1]
        gbr = m[2]
        Enet = m[3]
        lgb = m[4]
        las2 = m[5]
        ger_predict = ger.predict(X_score)
        las_predict = las.predict(X_score)
        gbr_predict = gbr.predict(X_score)
        Enet_predict = Enet.predict(X_score)
        lgb_predict = lgb.predict(X_score)
        X_stack = pd.DataFrame({"A": []})
        X_stack = pd.concat([X_stack, pd.DataFrame(ger_predict), pd.DataFrame(las_predict), pd.DataFrame(gbr_predict),
                             pd.DataFrame(Enet_predict), pd.DataFrame(lgb_predict)], axis=1)
        X_stack = np.array(X_stack)
        X_stack = np.delete(X_stack, 0, 1)
        scores_fin = scores_fin * (las2.predict(X_stack))
    scores_fin = scores_fin ** (1 / nF)
