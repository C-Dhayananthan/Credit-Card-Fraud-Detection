#!/usr/bin/env python
# coding: utf-8

# In[2]:


def clf_metrics(y_true,y_predicted,f_beta = 1):
    from sklearn.metrics import (confusion_matrix,
                                 fbeta_score,
                                 recall_score,
                                 classification_report,
                                 accuracy_score)
    metrices = [confusion_matrix , fbeta_score , recall_score , classification_report , 
                accuracy_score]
    met_names = {1:'confusion_matrix'  , 2:'fbeta_score' , 3:'recall_score', 4:"classification_report",
                 5 : 'accuracy_score'}
    n = 0
    for met in metrices:
        n+=1 
        if met_names[n] == 'fbeta_score':
            print(met_names[n]+' : ',met(y_true,y_predicted,beta = f_beta))
            print('\n')
        
        elif met_names[n] == "classification_report":
            print(met_names[n]+' : ')
            print(met(y_true,y_predicted))
            print('\n')
        else :
            print(met_names[n]+' : ',met(y_true,y_predicted))
            print('\n')


# In[ ]:




