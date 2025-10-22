import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.special import expit

PX_SIZE = 784
N_CL = 26
np.random.seed(42)


def load_data():
    FILE_NAME = 'emnist-letters-train.csv'
    try:
        print(f"Завантаження елементів {FILE_NAME}")

        df = pd.read_csv(FILE_NAME, header=None)
        Y_raw = df.iloc[:, 0].values
        X_raw = df.iloc[:, 1:].values

        if Y_raw.min() >= 1:
            Y_raw = Y_raw - 1

        print(f"Завантажено {X_raw.shape[0]} елементів")
        return X_raw, Y_raw

    except FileNotFoundError:
        print(f"Файл '{FILE_NAME}' не знайдено")
        N_SAMPLES = 10000
        Y_raw = np.random.randint(0, N_CL, N_SAMPLES)
        X_raw = np.random.rand(N_SAMPLES, PX_SIZE) * 255
        return X_raw, Y_raw


def one_hot_encode(y):
    Y_oh = np.zeros((y.size, N_CL))
    Y_oh[np.arange(y.size), y.astype(int)] = 1
    return Y_oh


X_data, Y_labels = load_data()
Y_oh_all = one_hot_encode(Y_labels)

X_train_raw, X_test_raw, Y_train_oh, Y_test_oh = train_test_split(X_data, Y_oh_all, test_size=0.1, random_state=42)

X_tr_raw, X_val_raw, Y_tr_oh, Y_val_oh = train_test_split(X_train_raw, Y_train_oh, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

_, _, _, Y_test_single = train_test_split(X_data, Y_labels, test_size=0.1, random_state=42)
Y_val_single = np.argmax(Y_val_oh, axis=1)

print(f"Px size: {X_tr.shape[1]}")
print("-" * 50)

class MLP:
    def __init__(self, in_sz, h_sz, out_sz=N_CL, lr=0.05, mom=0.9):
        self.lr, self.mom = lr, mom

        self.W1 = np.random.uniform(-np.sqrt(6 / (in_sz + h_sz)), np.sqrt(6 / (in_sz + h_sz)), (in_sz, h_sz))
        self.b1 = np.zeros((1, h_sz))
        self.W2 = np.random.uniform(-np.sqrt(6 / (h_sz + out_sz)), np.sqrt(6 / (h_sz + out_sz)), (h_sz, out_sz))
        self.b2 = np.zeros((1, out_sz))

        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

        self.loss_hist, self.acc_hist = [], []

    def activation(self, x):
        return expit(x)

    def deriv_activation(self, x):
        s = self.activation(x)
        return s * (1 - s)

    def output_func(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss_func(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.output_func(self.Z2)
        return self.A2

    def backward(self, X, Y_true, Y_pred):
        m = X.shape[0]

        dZ2 = Y_pred - Y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.deriv_activation(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.vW2 = self.mom * self.vW2 - self.lr * dW2
        self.vb2 = self.mom * self.vb2 - self.lr * db2
        self.vW1 = self.mom * self.vW1 - self.lr * dW1
        self.vb1 = self.mom * self.vb1 - self.lr * db1
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

    def predict(self, X):
        Y_probs = self.forward(X)
        return np.argmax(Y_probs, axis=1)

    def evaluate(self, X, Y_true_oh):
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y_true_oh, axis=1)
        return accuracy_score(Y_true, Y_pred)

    def train(self, X_tr, Y_tr, X_val, Y_val, epochs=100):
        for epoch in range(1, epochs + 1):
            Y_pred_tr = self.forward(X_tr)
            loss = self.loss_func(Y_tr, Y_pred_tr)
            self.loss_hist.append(loss)

            self.backward(X_tr, Y_tr, Y_pred_tr)

            acc = self.evaluate(X_val, Y_val)
            self.acc_hist.append(acc)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Епоха {epoch:03}/{epochs} | Втрати: {loss:.4f} | Точність (вал): {acc:.4f}")


print("\n--- навчання ----")
H_SIZE = 400
L_RATE = 0.05
EPOCHS = 100

model = MLP(PX_SIZE, H_SIZE, lr=L_RATE)
model.train(X_tr, Y_tr_oh, X_val, Y_val_oh, epochs=EPOCHS)

final_acc = model.evaluate(X_test, Y_test_oh)
print("-" * 50)
print(f"Точність: {final_acc:.4f}")
print("-" * 50)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model.loss_hist, label='Втрати')
plt.title('Динаміка втрат під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model.acc_hist, label='Точність')
plt.title('Динаміка точності на валідації')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.grid(True)
plt.show()

Y_pred_test = model.predict(X_test)
labels_map = {i: chr(ord('A') + i) for i in range(N_CL)}
cm = confusion_matrix(Y_test_single, Y_pred_test)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=False, cmap='Blues', fmt='.2f', xticklabels=labels_map.values(), yticklabels=labels_map.values())
plt.ylabel('Справжня мітка')
plt.xlabel('Передбачена мітка')
plt.title('Нормалізована матриця помилок (Помилки)')
plt.show()

np.fill_diagonal(cm, 0)

misclassification_counts = []
for i in range(N_CL):
    for j in range(N_CL):
        if i != j and cm[i, j] > 0:
            misclassification_counts.append({
                'Actual': labels_map[i],
                'Predicted': labels_map[j],
                'Count': cm[i, j]
            })


