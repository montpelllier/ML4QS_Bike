from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

# 加载数据
DATA_PATH = Path('./results/')
DATASET_FNAME = 'dataset1_result_feature.csv'
RESULT_FNAME = 'classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/classification/')

# we declare the parameters we'll use in the algorithms.
N_FEATURE_SELECTION = 25

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

# 划分特征和目标变量
X = dataset.drop(
    columns=['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate'])
y = dataset[['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate']].copy()

# 将多标签转换为单标签
y['label'] = y.idxmax(axis=1)
label_encoder = LabelEncoder()
y['label'] = label_encoder.fit_transform(y['label'])

# 确保数据按时间顺序分割，假设数据已经按时间排序
# train_size = int(0.7 * len(dataset))  # 使用70%的数据作为训练集
# X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
# y_train, y_test = y['label'].iloc[:train_size], y['label'].iloc[train_size:]
# print(len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y['label'], test_size=0.3, random_state=42)

# 处理训练集中的样本不平衡
train_data = X_train.copy()
train_data['label'] = y_train
train_data_balanced = pd.concat([resample(train_data[train_data['label'] == label],
                                          replace=True,  # 上采样
                                          n_samples=train_data['label'].value_counts().max(),
                                          random_state=42) for label in train_data['label'].unique()])

X_train_balanced = train_data_balanced.drop(columns=['label'])
y_train_balanced = train_data_balanced['label']

# 定义多分类模型
model = RandomForestClassifier()

# 递归特征选择在平衡后的训练集上进行
rfe = RFE(model, n_features_to_select=N_FEATURE_SELECTION)
rfe.fit(X_train_balanced, y_train_balanced)

# 查看选择的特征
selected_features = X_train_balanced.columns[rfe.support_]
print("Selected Features:", selected_features)

# 转换训练集和测试集并选择特征
X_train_selected = X_train_balanced[selected_features]
X_test_selected = X_test[selected_features]

# 定义参数网格进行网格搜索
param_grid = {
    'n_estimators': [30, 50, 70],
    'max_depth': [None, 1, 3, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# 在选择的特征上重新训练模型
grid_search.fit(X_train_selected, y_train_balanced)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_selected)

# 将预测结果转换为原始标签格式
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# 评估模型性能
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print("Model Accuracy with Selected Features and Grid Search:", accuracy)
print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))
