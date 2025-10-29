import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay

data = pandas.read_csv('transport_emission_check.csv')
print(data)

x = data[['Vehicle_Age', 'Engine_Size', 'Mileage_per_Year', 'Previous_Failures']]
y = data['Emission_Test_Result']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 10)
model = LogisticRegression()
model.fit(x_train,y_train)
predicted_y = model.predict(x)
print(predicted_y)

print("Accuracy:",accuracy_score(y_test,model.predict(x_test)))
print(confusion_matrix(y_test,predicted_y))
print(classification_report(y_test,predicted_y))

cm = confusion_matrix(y_test,predicted_y)
chart = ConfusionMatrixDisplay(confusion_matrix=cm)
chart.plot(cmap='Blues')
pyplot.title('Confusion Matrix')
pyplot.show()
