{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6f7a210f",
   "metadata": {},
   "source": [
    "# Smart Lender - Loan Eligibility Prediction System\n",
    "## Exploratory Data Analysis\n",
    "\n",
    "This notebook explores the loan dataset, highlights key patterns, and summarizes each visualization with a short conclusion."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b49363db",
   "metadata": {},
   "source": [
    "## 1. Import Libraries and Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1819dc3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('Set2')\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.width', 120)\n",
    "\n",
    "data_path = Path('..') / 'dataset' / 'loan_data.csv'\n",
    "if not data_path.exists():\n",
    "    raise FileNotFoundError(f'Dataset not found at: {data_path.resolve()}')\n",
    "\n",
    "df = pd.read_csv(data_path)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76db45cc",
   "metadata": {},
   "source": [
    "## 2. Dataset Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f6dc0c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Shape:', df.shape)\n",
    "print('\\nColumn Names:')\n",
    "print(df.columns.tolist())\n",
    "print('\\nDataset Info:')\n",
    "df.info()\n",
    "print('\\nDescriptive Statistics:')\n",
    "display(df.describe(include='all').T)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d77f6247",
   "metadata": {},
   "source": [
    "**Conclusion:** The dataset overview shows the number of records, feature types, and basic statistics, which helps identify numeric and categorical columns before deeper analysis."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b57126d",
   "metadata": {},
   "source": [
    "## 3. Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2742cafc",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_values = df.isnull().sum().sort_values(ascending=False)\n",
    "missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)\n",
    "missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})\n",
    "display(missing_df[missing_df['Missing Values'] > 0])\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='mako')\n",
    "plt.title('Missing Value Heatmap')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4aa07f9",
   "metadata": {},
   "source": [
    "**Conclusion:** Missing values should be handled carefully because they can bias model training and affect loan approval predictions if left untreated."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22e80c61",
   "metadata": {},
   "source": [
    "## 4. Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c616044",
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_df = df.select_dtypes(include=[np.number])\n",
    "plt.figure(figsize=(12, 8))\n",
    "sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)\n",
    "plt.title('Correlation Heatmap')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74c833ee",
   "metadata": {},
   "source": [
    "**Conclusion:** Correlation values reveal which numeric variables move together and help identify potentially useful features for the classification model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5940cf88",
   "metadata": {},
   "source": [
    "## 5. Loan Approval Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be8b6a1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "target_column = 'Loan_Status' if 'Loan_Status' in df.columns else df.columns[-1]\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.countplot(data=df, x=target_column, palette='Set2')\n",
    "plt.title('Loan Approval Distribution')\n",
    "plt.xlabel('Loan Status')\n",
    "plt.ylabel('Count')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a550ae9",
   "metadata": {},
   "source": [
    "**Conclusion:** The target distribution shows whether the dataset is balanced or skewed, which is important for choosing evaluation metrics and model strategy."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1bbd6392",
   "metadata": {},
   "source": [
    "## 6. Gender Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc58e5c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'Gender' in df.columns:\n",
    "    plt.figure(figsize=(8, 5))\n",
    "    sns.countplot(data=df, x='Gender', palette='Set2')\n",
    "    plt.title('Gender Distribution')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Gender column not found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e07d7e63",
   "metadata": {},
   "source": [
    "**Conclusion:** Gender distribution helps assess representation across applicants and can be compared with approval patterns for fairness insights."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1188a56a",
   "metadata": {},
   "source": [
    "## 7. Education Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f367a38c",
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'Education' in df.columns:\n",
    "    plt.figure(figsize=(8, 5))\n",
    "    sns.countplot(data=df, x='Education', palette='Set3')\n",
    "    plt.title('Education Distribution')\n",
    "    plt.xticks(rotation=15)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Education column not found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a472867c",
   "metadata": {},
   "source": [
    "**Conclusion:** Education levels may influence lending outcomes by reflecting applicant background stability and employability."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fbe47c3",
   "metadata": {},
   "source": [
    "## 8. Property Area Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "324e1a21",
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'Property_Area' in df.columns:\n",
    "    plt.figure(figsize=(8, 5))\n",
    "    sns.countplot(data=df, x='Property_Area', palette='Set2')\n",
    "    plt.title('Property Area Distribution')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Property_Area column not found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83326545",
   "metadata": {},
   "source": [
    "**Conclusion:** Property area can reflect locality-based lending risk and is often useful in understanding spatial trends in loan approvals."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c066646",
   "metadata": {},
   "source": [
    "## 9. Income Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e4260eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "income_columns = [col for col in ['ApplicantIncome', 'CoapplicantIncome'] if col in df.columns]\n",
    "if income_columns:\n",
    "    fig, axes = plt.subplots(1, len(income_columns), figsize=(14, 5))\n",
    "    if len(income_columns) == 1:\n",
    "        axes = [axes]\n",
    "    for axis, column in zip(axes, income_columns):\n",
    "        sns.histplot(df[column], kde=True, ax=axis, color='teal')\n",
    "        axis.set_title(f'{column} Distribution')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Income columns not found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "27db7a0f",
   "metadata": {},
   "source": [
    "**Conclusion:** Income distributions are usually right-skewed, so extreme values should be considered when building and validating the model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a326751",
   "metadata": {},
   "source": [
    "## 10. Loan Amount Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "822506d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'LoanAmount' in df.columns:\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    sns.histplot(df['LoanAmount'].dropna(), kde=True, color='coral')\n",
    "    plt.title('Loan Amount Distribution')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('LoanAmount column not found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8d62fd4",
   "metadata": {},
   "source": [
    "**Conclusion:** Loan amount distribution helps reveal whether most applicants request moderate loans or whether the dataset contains large-value outliers."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b09a782e",
   "metadata": {},
   "source": [
    "## 11. Credit History Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2467ed3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'Credit_History' in df.columns and target_column in df.columns:\n",
    "    plt.figure(figsize=(8, 5))\n",
    "    sns.countplot(data=df, x='Credit_History', hue=target_column, palette='Set1')\n",
    "    plt.title('Credit History vs Loan Approval')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Credit_History or target column not found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef611de1",
   "metadata": {},
   "source": [
    "**Conclusion:** Credit history is often one of the strongest approval signals, so its relationship with the target variable deserves special attention."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9fa223d5",
   "metadata": {},
   "source": [
    "## 12. Pair Plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4e269aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "pairplot_columns = [col for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] if col in df.columns]\n",
    "if len(pairplot_columns) >= 2:\n",
    "    sns.pairplot(df[pairplot_columns].dropna())\n",
    "    plt.suptitle('Pair Plots for Selected Numeric Features', y=1.02)\n",
    "    plt.show()\n",
    "else:\n",
    "    print('Not enough numeric columns available for pair plots.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03d11458",
   "metadata": {},
   "source": [
    "**Conclusion:** Pair plots make joint relationships and clusters visible, which helps identify feature interactions and nonlinear patterns."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef5fc7f3",
   "metadata": {},
   "source": [
    "## 13. Box Plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5b20f7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "box_columns = [col for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'] if col in df.columns]\n",
    "if box_columns:\n",
    "    plt.figure(figsize=(14, 5))\n",
    "    for index, column in enumerate(box_columns, start=1):\n",
    "        plt.subplot(1, len(box_columns), index)\n",
    "        sns.boxplot(y=df[column], color='skyblue')\n",
    "        plt.title(column)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('No box plot columns found in the dataset.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb98395e",
   "metadata": {},
   "source": [
    "**Conclusion:** Box plots show spread and skewness clearly, making them useful for spotting unusual income and loan amount values."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ed66b1c",
   "metadata": {},
   "source": [
    "## 14. Outlier Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "461e31ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "def detect_outliers_iqr(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:\n",
    "    summary_rows = []\n",
    "    for column in columns:\n",
    "        q1 = dataframe[column].quantile(0.25)\n",
    "        q3 = dataframe[column].quantile(0.75)\n",
    "        iqr = q3 - q1\n",
    "        lower_bound = q1 - 1.5 * iqr\n",
    "        upper_bound = q3 + 1.5 * iqr\n",
    "        outlier_count = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)].shape[0]\n",
    "        summary_rows.append({\n",
    "            'Column': column,\n",
    "            'Outliers': outlier_count,\n",
    "            'Lower Bound': lower_bound,\n",
    "            'Upper Bound': upper_bound\n",
    "        })\n",
    "    return pd.DataFrame(summary_rows)\n",
    "\n",
    "outlier_columns = [col for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'] if col in df.columns]\n",
    "if outlier_columns:\n",
    "    display(detect_outliers_iqr(df, outlier_columns))\n",
    "else:\n",
    "    print('No numeric columns available for outlier detection.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15dee049",
   "metadata": {},
   "source": [
    "**Conclusion:** Outlier detection helps identify extreme values that may distort model training and should be treated before final modeling."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b451b62",
   "metadata": {},
   "source": [
    "## 15. Final Conclusions"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce412625",
   "metadata": {},
   "source": [
    "The EDA indicates that loan approval is likely influenced by a combination of credit history, income level, loan amount, and applicant profile attributes.\n",
    "\n",
    "Key next steps include preprocessing categorical variables, handling skewed numeric features, and training a classification model with balanced evaluation metrics."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
