import pandas as pd
import numpy as np
from preprocessing import create_hour_feature, scale_amounts
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,average_precision_score,roc_auc_score
from dotenv import load_dotenv
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
load_dotenv()


# 1-> process data(hr feature, scaling, targets+feature separation, splitting)
# 2-> handle class imbalance
# 3-> train and save ml models
# 4-> main func


def preprocess_data(df):
    # creating hr features
    df=create_hour_feature(df)
    # separating features and targets
    X=df.drop("Class",axis=1)
    y=df["Class"]
    # splitting into training and test sets
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    # scaling amounts
    X_train,X_test=scale_amounts(X_train,X_test)
    return X_train,X_test,y_train,y_test

def handle_imbalance(X_train,y_train):
    smote=SMOTE(sampling_strategy=0.3,random_state=42)
    X_train_sm,y_train_sm=smote.fit_resample(X_train,y_train)
    return X_train_sm,y_train_sm

def get_models(y_train):
    return {
        # "logistic_regression":LogisticRegression(random_state=42,max_iter=1000),
        "XGBoost":XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),eval_metric='aucpr',#optimize for precision-recall AUC, not accuracy
                                random_state=42,n_estimators=300,max_depth=5,learning_rate=0.1),

        "LightGBM":LGBMClassifier(class_weight='balanced', #built-in imbalance handling
                                random_state=42,n_estimators=300,max_depth=5,learning_rate=0.1,verbosity=-1),

        "CatBoost":CatBoostClassifier(auto_class_weights='Balanced',#built-in imbalance handling
                                random_state=42,iterations=300,depth=5,learning_rate=0.1,verbose=0),

        "Decision Tree":DecisionTreeClassifier(class_weight='balanced',#weight classes inversely to frequency
                                max_depth=8,               # limit depth to prevent overfitting
                                min_samples_leaf=20,       # require at least 20 samples per leaf
                                random_state=42),

        "Random Forest":RandomForestClassifier(n_estimators=300, #number of trees in the forest
                                class_weight='balanced',
                                max_depth=10,
                                min_samples_leaf=10,
                                random_state=42,n_jobs=-1 ),
    }

def train_and_eval_model(X_train_sm,X_test,y_train_sm,y_test):
    save_dir=os.getenv("model_save_path")
    models=get_models(y_train_sm)
    best_model=None
    best_model_name=None
    best_ap=-1
    for name,model in models.items():
        model.fit(X_train_sm,y_train_sm)
        y_pred=model.predict(X_test)
        y_prob=model.predict_proba(X_test)[:,1]
        ap=average_precision_score(y_test,y_prob)
        auc=roc_auc_score(y_test,y_prob)
        print(f"{name}")
        print("Classification Report:\n",classification_report(y_test,y_pred))
        print("Accuracy Score: ",accuracy_score(y_test,y_pred))
        print(f"Average Precision Score: {average_precision_score(y_test,y_prob)}")
        print(f"ROC AUC Score: {roc_auc_score(y_test,y_prob)}")
        print("*"*30)
        if ap>best_ap:
            best_ap=ap
            best_model=model
            best_model_name=name
    os.makedirs(save_dir,exist_ok=True)
    safe_name=best_model_name.lower().replace(" ","_")
    save_path=os.path.join(save_dir,f"{safe_name}.pkl")
    joblib.dump(best_model,save_path)
    return best_model,best_model_name

def main():
    data=pd.read_csv(os.getenv("cleaned_df_path"))
    X_train,X_test,y_train,y_test=preprocess_data(data)
    # smote only for logistic regression, rest of the models handle imbalance internally
    # X_train_sm,y_train_sm=handle_imbalance(X_train,y_train)
    train_and_eval_model(X_train,X_test,y_train,y_test)

if __name__=="__main__":
    main()
