import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# 替换为你的 Arial.ttf 路径
# font_path = "/mnt/data1/tanshiqian/.fonts/Arial.ttf"  # Linux
# font_path = "C:/Windows/Fonts/Arial.ttf"  # Windows
# font_path = "/Library/Fonts/Arial.ttf"  # macOS

# font_prop = FontProperties(fname=font_path, size=10)
# font_prop = FontProperties(fname=font_path)
# 设置全局参数
plt.rcParams.update({
    # 'font.sans-serif': 'Arial',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.54, 2.5),  # 单栏矩形图
    'savefig.dpi': 600,
    'axes.titlesize': 10,
    # 'axes.labelweight': 'bold',
    # 'axes.titleweight': 'bold',
    # 'axes.titlepad': 5,
    # 'axes.labelpad': 5,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 3,
    # 'lines.markeredgewidth': 0.5,
    # 'lines.markerfacecolor': 'black',
    # 'lines.markeredgecolor': 'black',
})
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.size'] = 10
# plt.rcParams['font.sans-serif'] = 'Arial'  # 设置全局字体
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.set_cmap('viridis')


def parity_plots(model, X_train, y_train, cv, title=None, filename=None):

    # Initialize lists to store combined predictions and true values
    y_true_combined = []
    y_pred_combined = []

    # Loop through each fold's estimator and test indices
    for train_index, test_index in cv.split(X_train):
        # Get the test data and target
        X_test_fold = X_train[test_index]
        y_test_fold = y_train.values[test_index].ravel()
        
        # Predict on the test set
        # estimator = model.fit(X_train[train_index], y_train.values[train_index].ravel())
        y_test_pred = model.predict(X_test_fold)
        
        # Append predictions and true values to the combined lists
        y_true_combined.extend(y_test_fold)
        y_pred_combined.extend(y_test_pred)

    # Convert combined lists to numpy arrays
    y_true_combined = np.array(y_true_combined)
    y_pred_combined = np.array(y_pred_combined)

    # Calculate overall metrics
    r2 = r2_score(y_true_combined, y_pred_combined)
    mae = mean_absolute_error(y_true_combined, y_pred_combined)
    rmse = np.sqrt(mean_squared_error(y_true_combined, y_pred_combined))

    # Plot parity plot
    plt.figure(figsize=(3.6, 3.6))
    # plt.scatter(y_true_combined, y_pred_combined, alpha=0.7, label=f'Overall: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}')
    plt.scatter(y_true_combined, y_pred_combined, alpha=0.7)
    min_val = min(y_true_combined.min(), y_pred_combined.min())
    max_val = max(y_true_combined.max(), y_pred_combined.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parity Line')

    # Labels and legend
    # plt.xlabel('True value')
    # plt.ylabel('Predicted value')
    x_label = r'$C_{exp}\ (cm^3\ CO_2/cm^3\ PILs)$'
    y_label = r'$C_{pre}\ (cm^3\ CO_2/cm^3\ PILs)$'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title('Combined Parity Plot for Test Sets (5-Fold Cross-Validation)')
    if title:
        plt.title('Parity Plot for ' + title)
    else:
        plt.title('Parity Plot for Testing Set')
    plt.legend()
    plt.grid()
    if filename:
        plt.savefig(f'figs/{filename}_parity.tiff', format='tiff', dpi=600, bbox_inches='tight')
        plt.savefig(f'figs/{filename}_parity.pdf', format='pdf', bbox_inches='tight')
    plt.show()
