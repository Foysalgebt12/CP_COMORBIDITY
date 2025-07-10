import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from autogluon.tabular import TabularPredictor

# === Paths ===
input_file = r"ML_FINAL_F.csv"
output_dir = r"FINAL_AI_ML_2121"
os.makedirs(output_dir, exist_ok=True)

# === Load and Prepare Data ===
df = pd.read_csv(input_file)
df['Target'] = df['DepMap_essentiality_score'].apply(lambda x: 1 if x <= -0.5 else 0)

non_features = ['GEO_ID', 'Disease_Name', 'Regulation', 'Short_Name', 'Gene_Symbol', 'Target']
feature_cols = [col for col in df.columns if col not in non_features and not col.startswith("Unnamed")]
X = df[feature_cols]
y = df['Target']
X.fillna(X.median(), inplace=True)
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# === Random Forest Feature Selection ===
rf = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
rf.fit(X_scaled, y)
selector = SelectFromModel(rf, prefit=True)
selected_features = X.columns[selector.get_support()]
if len(selected_features) < 30:
    top_30_idx = np.argsort(rf.feature_importances_)[-30:]
    selected_features = X.columns[top_30_idx]

X_selected = X_scaled[selected_features]
X_selected['label'] = y
X_selected.to_csv(os.path.join(output_dir, "Selected_Features_With_Label.csv"), index=False)

# === Feature Importance Plot ===
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_imp[selected_features].plot(kind='barh')
plt.title("Top Selected Features (Random Forest Importance)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Feature_Importance_RF_Selected.png"), dpi=300)
plt.close()

# === AutoML Training (AutoGluon) ===
automl_dir = os.path.join(output_dir, "autogluon_model")
predictor = TabularPredictor(label='label', path=automl_dir).fit(
    X_selected, presets='medium_quality', fit_weighted_ensemble=True,
    ag_args_fit={'max_memory_usage_ratio': 0.5}, verbosity=2
)
leaderboard = predictor.leaderboard(silent=True)
leaderboard.to_csv(os.path.join(output_dir, "AutoML_Model_Performance.csv"), index=False)

# === SHAP Analysis ===
X_explain = X_selected.drop(columns='label')
best_model_name = leaderboard.iloc[0]['model']
best_model = predictor._trainer.load_model(best_model_name)

try:
    explainer = shap.TreeExplainer(best_model.model)
    shap_values = explainer.shap_values(X_explain)
    shap.summary_plot(shap_values, X_explain, show=False)
except:
    masker = shap.maskers.Independent(X_explain)
    explainer = shap.Explainer(best_model.predict, masker=masker)
    shap_values = explainer(X_explain)
    shap.summary_plot(shap_values, X_explain, show=False)

plt.savefig(os.path.join(output_dir, "SHAP_Summary_AutoML.png"), dpi=300)
plt.close()

# === Save Model and Predictions ===
predictor.save()
preds = predictor.predict(X_selected.drop(columns='label'))
preds.to_csv(os.path.join(output_dir, "Predictions.csv"), index=False)
joblib.dump(best_model, os.path.join(output_dir, "Best_Model_AutoML.joblib"))

# === Gene Table with URLs ===
top_100 = df.drop_duplicates(subset=['Gene_Symbol'])[['Gene_Symbol', 'Target']].head(100)
top_100['GeneCards_URL'] = top_100['Gene_Symbol'].apply(lambda g: f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={g}")
top_100['STRING_URL'] = top_100['Gene_Symbol'].apply(lambda g: f"https://string-db.org/network/human/{g}")
top_100.to_csv(os.path.join(output_dir, "TOP_100_CANDIDATE_GENES.csv"), index=False)

# === Cross-Validation Evaluation ===
models = {
    'RandomForest': rf,
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'MLP': MLPClassifier(max_iter=500)
}

results, roc_data, pr_data = [], [], []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

for name, model in models.items():
    aucs, recalls, accs, f1s, pr_aucs, tprs, prs = [], [], [], [], [], [], []
    for train_idx, test_idx in cv.split(X_scaled[selected_features], y):
        model.fit(X_scaled[selected_features].iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X_scaled[selected_features].iloc[test_idx])
        y_pred = np.argmax(y_prob, axis=1)
        y_true = y.iloc[test_idx]

        accs.append(accuracy_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = average_precision_score(y_true, y_prob[:, 1])
        pr_aucs.append(pr_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_pr = np.interp(mean_recall, recall[::-1], precision[::-1])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        prs.append(interp_pr)
        aucs.append(auc(fpr, tpr))

    results.append((name, np.mean(aucs), np.mean(pr_aucs), np.mean(f1s), np.mean(recalls), np.mean(accs)))
    roc_data.append(go.Scatter(x=mean_fpr, y=np.mean(tprs, axis=0), mode='lines', name=f"{name} (AUC={np.mean(aucs):.3f})"))
    pr_data.append(go.Scatter(x=mean_recall, y=np.mean(prs, axis=0), mode='lines', name=f"{name} (PR-AUC={np.mean(pr_aucs):.3f})"))

# === Save ROC and PR Curves ===
roc_data.append(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc = go.Figure(data=roc_data)
fig_roc.update_layout(title='Cross-Validated ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig_roc.write_html(os.path.join(output_dir, "CrossVal_ROC_Curves.html"))

pr_data.append(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', name='Random', line=dict(dash='dash')))
fig_pr = go.Figure(data=pr_data)
fig_pr.update_layout(title='Cross-Validated Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision')
fig_pr.write_html(os.path.join(output_dir, "CrossVal_PR_Curves.html"))

# === Save Model Comparison Table ===
results_df = pd.DataFrame(results, columns=['Model', 'ROC_AUC', 'PR_AUC', 'F1', 'Recall', 'Accuracy'])
results_df.to_csv(os.path.join(output_dir, "CrossVal_Results_Expanded.csv"), index=False)

# === Plot Each Metric ===
for metric in ['ROC_AUC', 'PR_AUC', 'F1', 'Recall', 'Accuracy']:
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Model'], results_df[metric], edgecolor='black')
    plt.title(f'Model Comparison - {metric}')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Model_Comparison_{metric}.png"), dpi=300)
    plt.close()

print("\n✅ Nature++ pipeline complete with essentiality target. Outputs saved to:", output_dir)
