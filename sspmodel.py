import pandas as pd
import numpy as np 
import datetime
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib 




# importing data:

datetime_format = "%Y%m%d"

df = pd.read_csv("C:\\Users\\charl\\Downloads\\Sales_Product_Details.csv", 
                 parse_dates=['Date'], 
                 date_parser=lambda x: pd.to_datetime(x, format=datetime_format))
df


df.columns


# handling datetime
# Extracting Days
df["Date_Day"]=df["Date"].dt.day
df.head()


df.drop(columns = ["Date","Customer_ID","Product_ID","Product_Category","Product_Line","Raw_Material","Region","Latitude","Longitude"],axis= 1, inplace = True )

df.info()
df.isnull().sum()



df = pd.get_dummies(df,drop_first= True,dtype=np.int64)
df.head()

# renaming columns:
df.rename(columns = {   'Product_Description_Casual Shirts': 'Casual_Shirts',
    'Product_Description_Coats': 'Coats',
    'Product_Description_Cycling Jerseys': 'Cycling_Jerseys',
    'Product_Description_Dress': 'Dress',
    'Product_Description_Formal Shirts': 'Formal_Shirts',
    'Product_Description_GolfShoes': 'GolfShoes',
    'Product_Description_Jeans': "Jeans",              
    'Product_Description_Knitwear':"Knitwear" ,         
    'Product_Description_Pants':"Pants",              
    'Product_Description_Polo Shirts':"Polo_Shirts",        
    'Product_Description_Pyjamas':"Pyjamas",           
    'Product_Description_Shorts':"Shorts",            
    'Product_Description_Suits':"Suits",   
    'Product_Description_Sweats': 'Sweats',
    'Product_Description_Ties': 'Ties',
    'Product_Description_Tshirts': 'Tshirts',
    'Product_Description_Underwear': 'Underwear'}, inplace = True) 



df.head()

df.info()
df.isnull().sum()



# Outlier Detection:
from scipy.stats import zscore

# Calculate z-scores
z_scores = np.abs(zscore(df))
z_score_threshold = 3

# Create a boolean mask
outlier_mask = (z_scores > z_score_threshold).any(axis=1)
print(outlier_mask)

# Rows containing outliers
outliers = df[outlier_mask]
print("Outliers:")
print(outliers)



# Remove outliers from the DataFrame
df_no_outliers = df[~outlier_mask]

# Display information about the DataFrame without outliers
print("DataFrame without outliers:")
print(df_no_outliers.head())
print(df_no_outliers.info())
print(df_no_outliers.isnull().sum())




# selecting Target variable:
x=df_no_outliers.drop("Sales_Revenue",axis=1)             # input
y=df_no_outliers["Sales_Revenue"]                      # target variable

print(x)
print(y)

print("x_shape:",x.shape,"y_shape:",y.shape)


# model selection:
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)

print("x_train_shape:",x_train.shape,"y_train_shape:",y_train.shape)
print("x_test_shape:",x_test.shape,"y_test_shape:",y_test.shape)



# XGBoost Regressor
from xgboost import XGBRegressor

def train_xgboost_model():
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    trained_model= xgb_reg.fit(x_train, y_train)
    return trained_model

# save the model:
filename = "finalized_modelSP1.sav"
trained_model = train_xgboost_model()
joblib.dump(trained_model,filename)

# load the trained model:
loaded_model = joblib.load(filename)

# Main code prediction:
if __name__ == "__main__":
    prediction = loaded_model.predict([[1,36.193364,3,1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]]) # 1st row
    print("Prediction:", prediction) 

