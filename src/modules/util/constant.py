from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class Features:

    features = ["type", "flags", "assigned_to", "creator", "description", "summary", "id", "creation_time", "last_change_time", "resolution"]
    summary = "summary"
    description = "description"


class ModelName:
    
    NB = "Naive Bayes"
    LG = "Logistic Regression"
    DT = "Decision Tree"
    RF = "Random Forest"
    GB = "Gradient Boosting"



class Model:

    NB = GaussianNB()
    LG = LogisticRegression(max_iter=5000)
    DT = DecisionTreeClassifier()
    RF = RandomForestClassifier(random_state=42)
    GB = GradientBoostingClassifier(random_state=0)