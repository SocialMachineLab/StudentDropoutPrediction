import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

#ADM
# data=pd.read_csv('data/ADM_clean.csv')
#ARQ
# data=pd.read_csv('data/ARQ_clean.csv')
#CSI
data=pd.read_csv('data/CSI_clean.csv')
print(data.columns)
X, y = data.iloc[:,:-1],data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
trained_xgb = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

trained_xgb.fit(X_train,y_train)

preds = trained_xgb.predict(X_test)

print(preds)

#ADM

# cols=['sit_final','semestre_recomendado','grau','tentativas','diff','semestre_do_aluno',
#       'cant','matricula','periodo',
#       'puntos_enem']

# cols=['matricula', 'mat_ano', 'mat_sem', 'periodo', 'ano', 'semestre',
#        'disciplina', 'semestre_recomendado', 'semestre_do_aluno',
#        'no_creditos', 'turma', 'sit_final', 'nome_professor', 'cep',
#        'puntos_enem', 'diff', 'tentativas', 'cant', 'nome_disciplina', 'grau']
#ARQ

# cols=['sit_final','semestre_recomendado','grau','tentativas','diff',
#       'semestre_do_aluno','mat_ano','matricula','periodo',
#       'ano','disciplina','mat_sem']



# CSI

# cols=['sit_final','semestre_recomendado','grau','tentativas','cep',
#       'semestre_do_aluno','no_creditos','matricula','periodo',
#       'ano','disciplina','truma']

cols=['ano_curriculo', 'cod_curriculo', 'matricula', 'mat_ano', 'mat_sem',
       'periodo', 'ano', 'semestre', 'disciplina', 'semestre_recomendado',
       'semestre_do_aluno', 'no_creditos', 'turma', 'sit_final',
       'nome_professor', 'cep', 'puntos_enem', 'diff', 'tentativas', 'cant',
       'identificador', 'nome_disciplina', 'grau']






# compute all the importance of features
importance_all = pd.DataFrame()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    importance = trained_xgb.get_booster().get_score(importance_type=importance_type)
    keys = list(importance.keys())
    values = list(importance.values())
    df_importance = pd.DataFrame(data=values, index=keys, columns=['importance_'+importance_type])
    importance_all = pd.concat([importance_all, df_importance],axis=1)
print(importance_all)

explainer = shap.TreeExplainer(trained_xgb) # get explainer
shap_values = explainer.shap_values(data[cols]) # get SHAP values from each instance of each feature
y_base = explainer.expected_value
# plt.tight_layout()

shap.summary_plot(shap_values, data[cols],feature_names=cols,plot_type="bar")
# plt.savefig('ADM_BAR_PLOT.png', format='png', dpi=1200, bbox_inches='tight')

plt.show()