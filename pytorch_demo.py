import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_temp,y_train,y_temp = train_test_split(X,y,test_size =0.2,random_state = 42,stratify = y)
X_val,X_test,y_val,y_test = train_test_split(X,y,test_size =0.5,random_state = 42,stratify = y)

#对齐分布 已到类似【-1，1】
mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0 )
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) /std

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

#定义模型
class MLP(nn.Module):
    def __init__ (self):
        super().__init__()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,3)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()

#损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

EPOCHS = 100
train_loss_history ,val_loss_history = [],[]
train_acc_history ,val_acc_history = [],[]

for epoch in range(EPOCHS):
    output = model(X_train)
    loss = criterion (output,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted = output.argmax(dim=1)
    train_acc = (predicted == y_train).float().mean().item()

with torch.no_grad():
    val_output = model(X_val)
    val_loss = criterion(val_output,y_val)
    val_predicted = val_output.argmax(dim = 1)
    val_acc = (val_predicted == y_val).float().mean().item()

    train_loss_history.append(loss.item())
    val_loss_history.append(val_loss.item())
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)


    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

#测试集
with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output,y_test)
    test_predicted = test_output.argmax(dim = 1)
    test_acc = (test_predicted == y_test).float().mean().item()

print("\n=== 最终模型性能评估 ===")
print(f"训练集 - Loss: {train_loss_history[-1]:.4f}, Accuracy: {train_acc_history[-1]:.4f}")
print(f"验证集 - Loss: {val_loss_history[-1]:.4f}, Accuracy: {val_acc_history[-1]:.4f}")
print(f"测试集 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
 
        


