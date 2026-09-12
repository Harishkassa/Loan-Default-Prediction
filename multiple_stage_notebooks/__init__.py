
import os
import pandas as pd
import seaborn as sns
import shap
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              roc_curve, confusion_matrix, brier_score_loss,
                              precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight

from datetime import datetime, timezone

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)

from fairlearn.postprocessing import ThresholdOptimizer
from scipy.stats import ks_2samp, chi2_contingency



