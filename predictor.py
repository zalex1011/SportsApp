import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Διαβάζουμε το CSV
data = pd.read_csv("matches.csv")

# Δημιουργία ετικέτας αποτελέσματος
def result_label(row):
    if row['HomeScore'] > row['AwayScore']:
        return 1
    elif row['HomeScore'] < row['AwayScore']:
        return 2
    else:
        return 0

data['Result'] = data.apply(result_label, axis=1)

# Κωδικοποίηση Over/Under
le_over = LabelEncoder()
data['OverUnderLabel'] = le_over.fit_transform(data['OverUnder'])

# Χαρακτηριστικά (features)
features = ['HomeForm','AwayForm','HomeRank','AwayRank','HomeAvgScore','AwayAvgScore','HomeH2HWin','AwayH2HWin']

X = data[features]
y_result = data['Result']
y_over = data['OverUnderLabel']

# Εκπαίδευση μοντέλου Random Forest για αποτέλεσμα
X_train, X_test, y_train, y_test = train_test_split(X, y_result, test_size=0.2, random_state=42)
model_result = RandomForestClassifier(n_estimators=100, random_state=42)
model_result.fit(X_train, y_train)

# Εκπαίδευση μοντέλου Random Forest για Over/Under
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X, y_over, test_size=0.2, random_state=42)
model_over = RandomForestClassifier(n_estimators=100, random_state=42)
model_over.fit(X_train_over, y_train_over)

print("Τα μοντέλα εκπαιδεύτηκαν επιτυχώς!")

# Παράδειγμα πρόβλεψης για πρώτο παιχνίδι
example = X.iloc[0].values.reshape(1, -1)
pred_result = model_result.predict(example)[0]
pred_over = model_over.predict(example)[0]

result_map = {0: "Draw", 1: "HomeWin", 2: "AwayWin"}
over_map = {i: label for i, label in enumerate(le_over.classes_)}

print(f"Πρόβλεψη αποτελέσματος: {result_map[pred_result]}")
print(f"Πρόβλεψη Over/Under: {over_map[pred_over]}")
