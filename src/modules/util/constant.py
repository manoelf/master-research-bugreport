from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import RandomOverSampler


class Features:

    features = ["type", "flags", "assigned_to", "creator", "description", "summary", "id", "creation_time", "last_change_time"]
    train_test_features = ['status', 'changes_status', 'changes_resolution']
    # 'status_RESOLVED', 'status_VERIFIED'
    summary = "summary"
    description = "description"


class DataShowMetrics:
    
    columns={"index": "Resolution", "label": "Total"}


class ModelName:
    
    NB = "Naive Bayes"
    LG = "Logistic Regression"
    DT = "Decision Tree"
    RF = "Random Forest"
    GB = "Gradient Boosting"


class Model:

    NB = GaussianNB()
    LG = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
    DT = DecisionTreeClassifier()
    RF = RandomForestClassifier(random_state=42)
    GB = GradientBoostingClassifier(random_state=0)
    CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    ROS = RandomOverSampler()
