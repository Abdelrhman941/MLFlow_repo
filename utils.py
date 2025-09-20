import os, warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
warnings.filterwarnings('ignore')

# ========================= [1] Data Preparation =========================
def load_and_clean_data(path: str):
    """Load dataset and drop unused features + filter outliers."""
    df = pd.read_csv(path)
    
    # Drop irrelevant columns
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    
    # Remove unrealistic ages > 80
    df = df[df["Age"] <= 80]
    return df

BASE_DIR  = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
df = load_and_clean_data(DATA_PATH)

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=45)

# ========================= [2] Data Processing Pipelines =========================
NUM_COLS   = ["Age", "CreditScore", "Balance", "EstimatedSalary"]
CAT_COLS   = ["Gender", "Geography"]
READY_COLS = list(set(X_train.columns) - set(NUM_COLS) - set(CAT_COLS))

# Numerical pipeline
num_pipeline = Pipeline(
    steps=[
        ("select", DataFrameSelector(NUM_COLS)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)

# Categorical pipeline
cat_pipeline = Pipeline(
    steps=[
        ("select", DataFrameSelector(CAT_COLS)),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", sparse_output=False)),
    ]
)

# Ready pipeline (binary/ordinal features already in good format)
ready_pipeline = Pipeline(
    steps=[
        ("select", DataFrameSelector(READY_COLS)),
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]
)

# Combine all into one
full_pipeline = FeatureUnion(
    transformer_list=[
        ("numerical", num_pipeline),
        ("categorical", cat_pipeline),
        ("ready", ready_pipeline),
    ]
)

# Apply transformations
X_train_final = full_pipeline.fit_transform(X_train)
X_test_final  = full_pipeline.transform(X_test)

# ========================= [3] Save Processed Data =========================
# As I did OHE, The column number may be vary
out_categ_cols = full_pipeline.named_transformers['categorical'].named_steps['encode'].get_feature_names_out(CAT_COLS)

X_train_final = pd.DataFrame(full_pipeline.transform(X_train), columns=NUM_COLS + list(out_categ_cols) + READY_COLS)
X_test_final  = pd.DataFrame(full_pipeline.transform(X_test), columns=NUM_COLS + list(out_categ_cols) + READY_COLS)

# Dump the X_test_processed
X_test_final.to_csv(os.path.join(os.getcwd(), 'Output', 'X_test_processed.csv'), index=False)