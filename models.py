#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def clf_models(X,y, std= True , random_state = 42 , test_size = 0.3,
               estimators_rf = 50 , f_beta = 1,pre_train = 0):
    """"
    Random forest,ADA,XG,log_reg
    std = True in default
    random_state = 42,test_size = 0.3
    estimators_rf = 50 , f_beta = 1(defaults)
    pre_train means metrices for train dataset in default - 0
    
    """
    #imports
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xg
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    
    ss = StandardScaler()
    
    #data_split
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=test_size, random_state=random_state)
    
    #random forest
    rfc = RandomForestClassifier()
    if std == True:
        rfc.fit(ss.fit_transform(X_train),y_train)
        rfc_predicted_test = rfc.predict(ss.fit_transform(X_test))
        rfc_predicted_train = rfc.predict(ss.fit_transform(X_train))
        
    else :
        rfc.fit(X_train,y_train)
        rfc_predicted_test = rfc.predict(X_test)
        rfc_predicted_train = rfc.predict(X_train)
    
    
    #XG BOOST
    xgb = xg.XGBClassifier(use_label_encoder =False)
    if std == True:
        xgb.fit(ss.fit_transform(X_train),y_train)
        xgb_predicted_test = xgb.predict(ss.fit_transform(X_test))
        xgb_predicted_train = xgb.predict(ss.fit_transform(X_train))
        
    else :
        xgb.fit(X_train,y_train)
        xgb_predicted_test = xgb.predict(X_test)
        xgb_predicted_train = xgb.predict(X_train)
        
    
    #ADA BOOST
    ada = AdaBoostClassifier()
    if std == True:
        ada.fit(ss.fit_transform(X_train),y_train)
        ada_predicted_test = ada.predict(ss.fit_transform(X_test))
        ada_predicted_train = ada.predict(ss.fit_transform(X_train))
    else :
        ada.fit(X_train,y_train)
        ada_predicted_test = ada.predict(X_test)
        ada_predicted_train = ada.predict(X_tain)
    
    
    #LOG_REG
    log = LogisticRegression()
    if std == True:
        log.fit(ss.fit_transform(X_train),y_train)
        log_predicted_test = log.predict(ss.fit_transform(X_test))
        log_predicted_train = log.predict(ss.fit_transform(X_train))
        
    else:
        log.fit(X_train,y_train)
        log_predicted_test = log.predict(X_test)
        log_predicted_train = log.predict(X_train)
        
    models_out_test = [rfc_predicted_test , ada_predicted_test , xgb_predicted_test , log_predicted_test]
    models_names = {1:'RandomForestClassifier',2:'AdaBoostClassifier',3:'xgboost',4:'LogisticRegression'}
    models_out_train = [rfc_predicted_train , ada_predicted_train , xgb_predicted_train , log_predicted_train]
    
    #METRICES
    #imports
    from sklearn.metrics import confusion_matrix,fbeta_score,classification_report,accuracy_score

    if pre_train == 0:
        count = 0
        for i in models_out_test:
            count+=1
            print("### -------------------- Predicted by {} for test dataset---------------- ##".format(models_names[count]))
            print(classification_report(y_test,i))
            print("fbeta_score: ", fbeta_score(y_test,i,beta = f_beta))
            print("accuracy_score: ",accuracy_score(y_test,i))
            print("\n")
    else:
        n = 0
        for i in models_out_train:
            n+=1
            print("### -------------------- Predicted by {} for train dataset---------------- ##".format(models_names[count]))
            print(classification_report(y_train,i))
            print("fbeta_score: ", fbeta_score(y_train,i,beta = f_beta))
            print("accuracy_score: ",accuracy_score(y_train,i))
            print("\n")

