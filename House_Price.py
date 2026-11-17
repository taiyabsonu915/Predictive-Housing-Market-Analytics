import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb


df = pd.read_csv(r'C:\Users\HP\Documents\pyspider\projects\House_price_prediction\House_price.csv')


# print("Dataset Shape:", df.shape)

# print(df.head())
# Missing values checking
# print("\nMissing Values:")
# print(df.isnull().sum())

cat_cols = df.select_dtypes(include=['object']).columns
# print("Categorical Columns:")
for col in cat_cols:
    # print(f"{col}: {df[col].nunique()} unique values")
    pass

# 1.  Text columns label encoding by  LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("encoding starting by sklearn \n")

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Train/Test mein split ho gaya:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 3.Log Transformation  Target variable (Price)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# 4. Model created and  training on  log-transformed target
xgb_log_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
#  XGBoost training start with Log transformation
xgb_log_model.fit(X_train, y_train_log)


y_pred_log = xgb_log_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)

# 6. Final Accuracy aur MAE
r2_log = r2_score(y_test, y_pred_original)
mae_log = mean_absolute_error(y_test, y_pred_original)

# print(f"\n🔥  (Accuracy) after Log Transform : {r2_log:.4f}")
# print(f"📉 New MAE: ₹ / $ {mae_log:.2f}")

suburb_mapping = y_train.groupby(X_train['Suburb']).mean().to_dict()
overall_mean_price = y_train.mean()


def simple_predict(rooms, distance, landsize, suburb):
    suburb_code = suburb_mapping.get(suburb, overall_mean_price)
    
    data = {
        'Suburb': suburb_code, 
        'Rooms': rooms,
        'Distance': distance,
        'Landsize': landsize,
        'Type': 'h', 
        'Method': 'S', 
        'SellerG': 'Nelson', 
        'Regionname': 'Southern Metropolitan',
        'Propertycount': 4000, 
        'CouncilArea': 'Yarra', 
        'Bedroom2': rooms, 
        'Bathroom': 2, 
        'Car': 1, 
        'BuildingArea': 120.0
    }
    
    pred_df = pd.DataFrame([data])
    pred_df = pred_df[X_train.columns] 
    
    for col in ['Type', 'Method', 'SellerG', 'Regionname', 'CouncilArea']:
        if col in pred_df.columns:
            pred_df[col] = 0
            
    log_prediction = xgb_log_model.predict(pred_df)
    final_price = np.expm1(log_prediction[0])
    
    return final_price

round=True

while round:
    rooms = int(input("give how may rooms: "))
    distance = float(input("give how many distance: "))
    landsize = float(input("give the landsize: "))
    suburb = input("give the the suburb: ")
    price = simple_predict(rooms, distance, landsize, suburb)
    print(f"🏠 Predicted House Price: ₹ / $ {price:,.2f}")