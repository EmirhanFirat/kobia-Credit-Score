from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

drop_cols=["id"]

cat_cols=["sex","education","marriage"]
#ordinal/sıralı
ord_cols=["pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]

num_cols=["limit_bal", "age",
            "bill_amt1","bill_amt2","bill_amt3",
            "bill_amt4","bill_amt5","bill_amt6",
            "pay_amt1","pay_amt2","pay_amt3",
            "pay_amt4","pay_amt5","pay_amt6"]


#numeric sütunlar için pipeline
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#kategorik sütunlar için pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=[
        [-2, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
        ]*6))
])

ColumnTransformer(transformers=[
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
    ("ord", ordinal_pipeline, ord_cols)
],  remainder="drop")


def build_pipeline(classifier)-> Pipeline:
    """
    preprocesser modeli tek bir pipeline'a bağlar.
    bu şekilde Dependency Injection yaparak istediğimiz modeli pipeline'a verebilirim.
    örnek olarak:
    Xgb_pipeline = build_pipeline(XGBClassifier())
    light_pipeline = build_pipeline(LGBMClassifier())
    SOLID'in "O" harfi: Open/Closed Principle-> uygulamış oldum.
    Args:
        classifier: eğitilecek model(Xgboost,LightGBM)
    Returns:
        tam pipeline
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
        ("ord", ordinal_pipeline, ord_cols)
        ],  remainder="drop")
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
