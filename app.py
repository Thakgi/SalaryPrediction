import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


def ohe(df, cat_vars):
    df_processed = pd.get_dummies(df, prefix_sep="__", columns=cat_vars)
    return df_processed

def rem_addition_cols(df_test_processed,cat_vars,cat_dummies,df_processed_columns):
    for col in df_test_processed.columns:
        if("__" in col) and (col.split("__"[0]) in cat_vars) and col not in cat_dummies:
            print("Removing additional feature {}".format(col))
            df_test_processed.drop(col, axis=1, inplace = True)
        else:
            print("Nothing to remove")

    for col in cat_dummies:
        if col not in df_test_processed.columns and col != ['salary']:
            print("Adding missing feature {}".format(col))
            df_test_processed[col] = 0
    df = df_test_processed[df_processed_columns]
    return df

    
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    df_new = pd.DataFrame(columns= ['jobId', 'companyId','jobType','degree','major','industry','yearsExperience','milesFromMetropolis'])
    
    jobId = "JOB1362685407680"
    companyId = "COMP21"
    jobType = request.form.get('jobType')
    degree = request.form.get('degree')
    major = request.form.get('major')
    industry = request.form.get('industry')
    yearsExperience = request.form.get('yearsExperience')
    milesFromMetropolis = request.form.get('milesFromMetropolis')
    
    cat_vars=['jobType','degree','major','industry']
    num_vars=['yearsExperience','milesFromMetropolis']
    df1 = pd.DataFrame(data=[[jobId,companyId,jobType,degree,major,industry,yearsExperience,milesFromMetropolis]],columns=['jobId', 'companyId','jobType','degree','major','industry','yearsExperience','milesFromMetropolis'])
    df = pd.concat([df_new,df1], axis=0)
    df_new = ohe(df,cat_vars)

    df_ = rem_addition_cols(df_new,cat_vars,new_dict_cat_dummies,new_dict_processed_columns)
    prediction = model.predict(df_)
     
    return render_template('index.html', prediction_text='Salary predicted is : $ {}'.format(prediction))


# main application
if __name__ == '__main__':
    model = pickle.load(open('model.pkl', 'rb'))
    infile_processed_columns = open('processed_columns','rb')
    new_dict_processed_columns = pickle.load(infile_processed_columns)
    infile_processed_columns.close()

    infile_cat_dummies = open('cat_dummies','rb')
    new_dict_cat_dummies = pickle.load(infile_cat_dummies)
    infile_cat_dummies.close()
    new_dict_cat_dummies
    app.run(debug=True)
   