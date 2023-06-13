import pandas as pd
import numpy as np
import streamlit as st
import re
import joblib
import pickle
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import requests
shap.initjs()

##########################################################################################################################
###                                                 Titre                                                              ###
##########################################################################################################################
st.title(" Customer Dashboard : Loans")
st.write('')
st.write('')
st.write('')
##########################################################################################################################


##########################################################################################################################
###                                                 Fonctions utilisées                                                ###
##########################################################################################################################
### to reduce memory usage
# def reduce_mem_usage(df):
#     """ iterate through all the columns of a dataframe and modify the data type
#         to reduce memory usage.
#     """
#     start_mem = df.memory_usage().sum() / 1024 ** 2
#     #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
#
#     for col in df.columns:
#         col_type = df[col].dtype
#
#         if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#         elif 'datetime' not in col_type.name:
#             df[col] = df[col].astype('category')
#
#     #end_mem = df.memory_usage().sum() / 1024 ** 2
#     #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#     #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
#
#     return df
#
###  Plot des variables pour analyse univarié
def plot_frequency(df, client_id, vars_, nrow, ncol):
    colors = [ '#32CD32', 'red']
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 6 * nrow))
    axes = np.ravel(axes)

    for i, feature in enumerate(vars_):
        ax = axes[i]
        sns.histplot(data=df, x=feature, hue='TARGET', kde=True,palette=colors,ax=ax, binwidth=0.03 )
        #sns.displot(df,x=feature,hue='TARGET',kde=True,color=colors[i],ax=ax)
        ax.set_ylabel('Frequency plot', fontsize=16)
        ax.set_xlabel(feature, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        client = df[feature][df['SK_ID_CURR'] == client_id].values[0]
        ymin, ymax = ax.get_ylim()
        y_position = ymax * 0.85
        ax.axvline(client, c='black', linewidth=1.5, alpha=0.6, label='Client_position', linestyle="--")
        ax.text(client, y_position, "Client", c='black', ha='right', va='baseline', rotation=90, fontsize=18)
        titre = "Distribution of the feature: " + feature + " & Positioning of client " + str(client_id)
        ax.set_title(titre, fontsize=16)
        ax.legend().remove()

    legend_labels = ['Target 0', 'Target 1']
    legend_elements = [plt.Line2D([0], [0], color=c, marker='', linestyle='-') for c in colors]
    fig.legend(legend_elements, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig, use_container_width=True)


def scat_plot_px(df, var1, var2, color):
    #fig = plt.figure(figsize=(6,6))
    fig = px.scatter(df, x=var1, y=var2, color_continuous_scale='rdylgn_r', color=color)

    # Ajouter l'individu spécifique
    x_client = df[var1].loc[df['SK_ID_CURR'] == int(var_code)].values[0]
    y_client = df[var2].loc[df['SK_ID_CURR'] == int(var_code)].values[0]

    fig.add_trace(go.Scatter(x=[x_client], y=[y_client], mode='markers', marker=dict(color='black', size=15), name='Client '+str(var_code)))
    # Enlever la légende "target"
    fig.update_traces(showlegend=False, selector=dict(name=color))
    # Déplacer la légende du client i
    fig.update_layout(legend=dict(x=0.99, y=1.05))  # Nouvelle position de la légende
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")


def scat_plot_sns(df, var1, var2, color):
    plt.figure(figsize=(12,8))
    # Créer le scatter plot avec Seaborn
    palette_0_1 = sns.color_palette(['green','red'])
    sns.scatterplot(data=df, x=var1, y=var2, hue=color, palette=palette_0_1)
    #sns.scatterplot(data=df, x=var1, y=var2, hue=color, palette='RdYlGn_r')

    # Ajouter l'individu spécifique en orange
    x_client = df[var1].loc[df['SK_ID_CURR'] == int(var_code)].values[0]
    y_client = df[var2].loc[df['SK_ID_CURR'] == int(var_code)].values[0]
    plt.scatter(x_client, y_client, color='orange', s=100, label='Client '+str(var_code))

    # Enlever la légende "target"
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [label for label in labels if label != color]

    # Afficher la légende
    plt.legend(handles, labels, fontsize=16)

    # Définir les limites des axes
    plt.xlim(df[var1].min(), df[var1].max())
    plt.ylim(df[var2].min(), df[var2].max())
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(var1, fontsize=16)
    plt.ylabel(var2, fontsize=16)
    # Diminuer la taille de la figure
    #plt.gcf().set_size_inches(12, 8)

    # Afficher la figure
    st.pyplot()


##########################################################################################################################



##########################################################################################################################
###                                          Loading data et des nodèles                                               ###
##########################################################################################################################

###  chargement de l'échantillon
#df_clients = reduce_mem_usage((pd.read_csv('df_1000.csv')))
df_clients = pd.read_csv('df_1000.csv')
### Nettoyage des colonnes du Dataframe
df_clients  = df_clients.rename(columns = lambda x:re.sub(' ', '_', x))

### chargement de l'explainer SHAP
#explainer = joblib.load('explainer.sav')
explainer = pickle.load(open('explainer_pkl.pkl', 'rb'))
st.set_option('deprecation.showPyplotGlobalUse', False)

###  Importer le modèle entrainé lightGBM
lgbm_clf = pickle.load(open('best_model_lgbm.pkl', 'rb'))
Threshold = 0.60
##########################################################################################################################



##########################################################################################################################
###                                                 Loading data clients                                               ###
##########################################################################################################################
### liste pour sélectionner un client
SK_ID_CURR = st.sidebar.selectbox('Please select a Client ID ', df_clients['SK_ID_CURR'])
data = {'SK_ID_CURR': str(SK_ID_CURR)}
input_df = pd.DataFrame(data, index=[0])
#st.write('input_df',input_df)

###recuperation de l'id du client
var_code = input_df['SK_ID_CURR'][0]
#st.write('var_code=', var_code)

###  Selection des informations du dataframe lié au client choisi
pred_client = df_clients[df_clients['SK_ID_CURR'] == int(var_code)] ############
#st.write(df_clients.head(2))
if len(pred_client)==0:
    st.write('No data found')
    st.stop()
##########################################################################################################################



##########################################################################################################################
###                   Prédiction (model/request) / Affichage Crédit accepté-refusé / Affichage gauge de score de risque                ###
##########################################################################################################################

# ### résultat du prédiction via le modele pickle
# y_pred_proba = lgbm_clf.predict_proba(pred_client.iloc[:, 2:])
# y_pred_proba_1 = y_pred_proba[0][1]
# y_pred= lgbm_clf.predict(pred_client.iloc[:, 2:])
# # st.write('y_pred_proba=', y_pred_proba)
# # st.write('y_pred=', y_pred)
#
# ### Calculer le rique du prêt
# risk = "{:,.0f}".format(y_pred_proba[0][1]*100)
# pred_0 = "{:,.0f}".format(y_pred_proba[0][0]*100)


### résultat du prédiction via requests API
#url = 'http://localhost:4000/predict/'+str(var_code)
### résultat du prédiction via heroku
url = 'https://api-p7-oc.herokuapp.com/predict/'+str(var_code)
#st.write(url)
response = requests.get(url)
result = response.json()
y_pred_proba_1 = result['predictions']
risk = y_pred_proba_1*100




### Affichage Crédit accepté/refusé
texte = " <span style='color:black;font-size:20px;'>Loan for client ID :  </span> " + str(SK_ID_CURR)
if int(risk) > Threshold*100:
    texte = texte + "  ┅┅➤ <span style='color:red;font-size:20px;'> ❌ REFUSED  </span>"
    st.write(texte,unsafe_allow_html=True)
else:
    texte = texte + "  ┅┅➤ <span style='color:green;font-size:20px;'> ✅ APPROVED </span>"
    st.write(texte,unsafe_allow_html=True)

### Affichage gauge de score de risque
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = y_pred_proba_1,#y_pred_proba[0][1],
    mode = "gauge+number+delta",
    title = {'text': "Risk of Failure",'font': {'size': 30}},
    delta = {'reference': Threshold,
             'increasing':{'color':'red'},
             'decreasing':{'color':'green'}},
    gauge = {'axis': {'range': [None, 1]},
             'bar':{'color': "black"},
             'steps' : [
                 {'range': [0, Threshold], 'color': "green"},
                 {'range': [Threshold, 1], 'color': "red"}],
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': Threshold}}))

st.plotly_chart(fig, use_container_width=True)
##########################################################################################################################



# ##########################################################################################################################
# ###                  Affichage des informations détaillée du client sélectionné                                        ###
# ##########################################################################################################################
# with st.expander("Detailed customer information :", expanded=False):
#     st.write("Here you can see the detailed information of the customer " + var_code)
#     st.write(pred_client.iloc[:,2:].T)
# ##########################################################################################################################
#


##########################################################################################################################
###                                                       Analyse shapely                                              ###
##########################################################################################################################
df_clients_shap=df_clients.copy()
df_clients_shap.set_index('SK_ID_CURR', inplace = True)
df_clients_shap.drop(['TARGET','ypred1'], axis=1, inplace=True)
#st.write(df_clients_shap.head(2))
### récupération des shap_values de notre échantillon
#explainer = shap.Explainer(lgbm_clf, df_clients_shap, feature_names=df_clients_shap.columns)
shap_values = explainer(df_clients_shap)



###....................................... Analyse shapely locale
### index de l'ID client renseigné
idx_clients_shap = df_clients_shap.index.get_loc(int(var_code))
colors = ['green','red']
#st.write(idx_clients_shap)

### feature importance locale
waterfall = shap.plots.waterfall(shap_values[idx_clients_shap])#,color=colors)
with st.expander("Details of the decision", expanded=False):
    st.pyplot(waterfall)
    st.write("<span style='color:Crimson;'> Factors that expose the client to the risk of loan default </span>", unsafe_allow_html=True)
    st.write("<span style='color:DodgerBlue;'> Criteria that increase the client's likelihood of loan repayment. </span>", unsafe_allow_html=True)

###.............................................. Analyse shapely globale
#feature importance globale
summary_plot = shap.summary_plot(shap_values, max_display=10,color=colors)
with st.expander("Decision criteria of the algorithm"):
    st.pyplot(summary_plot)
    st.write('This graph illustrates the top 10 features that carry the most weight in all algorithmic decisions.')

# ##............................................. Récuperation de 10 features les plus importantes
#
#
# feature_names = shap_values.feature_names
# shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
# vals = np.abs(shap_df.iloc[idx_clients_shap].values)
# shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
# shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
# top_ten = shap_importance['col_name'].head(20).tolist()#reset_index(drop=True)
# top_ten = pd.DataFrame(top_ten)
##########################################################################################################################



##########################################################################################################################
###                              Univariate Analysis & Client Positioning                                              ###
##########################################################################################################################




with st.expander("More Analysis", expanded=True):
    st.write("<div id='shapley'><h6><span style='color:#0A1172;'>Analyse Client : "+var_code+"</h6></div></br>", unsafe_allow_html=True)
    with st.form("form1"):# = st.form(key="form")
        st.markdown("<div id='shapley'><h6>Univariate Analysis & Client Positioning</span></h6></div></br>", unsafe_allow_html=True)
        vars = ['EXT_SOURCE_MEAN', 'AMT_CREDIT', 'DAYS_BIRTH', 'INS_DPD_MEAN',
               'AMT_ANNUITY', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'AMT_GOODS_PRICE',
               'PREV_CNT_PAYMENT_MEAN','BUREAU_AMT_CREDIT_SUM_DEBT_MEAN', 'DAYS_EMPLOYED',
               'APPROVED_AMT_ANNUITY_MEAN', 'PREV_APP_CREDIT_PERC_MEAN',
               'DAYS_ID_PUBLISH', 'ACTIVE_DAYS_CREDIT_MEAN', 'INS_AMT_PAYMENT_MEAN', 'CODE_GENDER',
               'PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN', 'DAYS_LAST_PHONE_CHANGE',
               'INS_DAYS_ENTRY_PAYMENT_MEAN', 'INS_DBD_MEAN', 'FLAG_OWN_CAR',
               'BUREAU_AMT_CREDIT_SUM_MEAN', 'INS_DAYS_INSTALMENT_MEAN','PREV_AMT_DOWN_PAYMENT_MEAN']
        col_selected = st.multiselect('Choose one or more features :', vars)#df_clients.columns)
        submit = st.form_submit_button(label="submit")
        with st.spinner('Loading data'):
                if submit:
                    if (len(col_selected) == 0):
                        st.error('❌ Sélectionner au moins une variable')
                        st.stop()
                    elif (len(col_selected)>0):
                        #st.write(col_selected)
                        plot_frequency(df_clients,int(var_code),col_selected, nrow=len(col_selected), ncol=1)

    with st.form("form2"):
        st.markdown("<div id='shapley'><h6>Bivariate analysis</h6></div></br>", unsafe_allow_html=True)
        # all_continuous_features = df_clients_shap.select_dtypes(exclude=['object','bool']).columns.to_list()
        # col_selected_top10 = [x for x in all_continuous_features if x in top_ten]
        col_selected2 = st.multiselect('Choose 2 features :', vars)
        # var_1 = st.selectbox('1st Feature :',top_ten)
        # list_2=top_ten.drop(top_ten[top_ten['col_name']==var_1].index)
        # var_2 = st.selectbox('2nd Feature :',list_2)
        submit2 = st.form_submit_button(label="submit")
        with st.spinner('Loading data'):
                 if submit2:
                     if (len(col_selected2) != 2 ):
                         st.error('❌ Sélectionner juste 2 variables')
                         st.stop()
                     elif (len(col_selected2) == 2):
                         scat_plot_px(df_clients, col_selected2[0], col_selected2[1], color="ypred1")#,title=titre)
                         #scat_plot=px.scatter(df_clients, col_selected2[0], col_selected2[1], color="TARGET",color_continuous_scale='rdylgn_r')#,title=titre)
                         # Ajout de l'individu i colorié en rouge
                         #scat_plot (pred_client, col_selected2[0], col_selected2[1], color="DodgerBlue")#,title=titre)
                         #st.plotly_chart(scat_plot, use_container_width='auto')

##########################################################################################################################
