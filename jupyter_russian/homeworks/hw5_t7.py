for par in logit_pipe_params['logit__C']:
    print('====',par,'====')
    logit_pipe.set_params(logit__C=par)
    logit_pipe.fit(X.values,y.values)
    proba = logit_pipe.predict_proba(X.values)
    #print('proba = ', proba)
    score = roc_auc_score(y,proba[:, 1])
    print('score = ', score, '=>', round(score, 3))
