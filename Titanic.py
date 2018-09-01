import pandas as pd
import numpy as np
import re

data = pd.read_csv("C:/Python/ML/train.csv", index_col='PassengerId')

# 1
sex_counts = data['Sex'].value_counts()
result = pd.DataFrame(sex_counts).T
result.to_csv("C:/Python/ML/solution.txt", index=None, header=None, sep=" ", line_terminator='')

# 2
surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
survsol = np.array([surv_percent])
np.savetxt("C:/Python/ML/solution.txt", survsol, fmt='%1.2f', newline='')

# 3
pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
psol = np.array([pclass_percent])
np.savetxt("C:/Python/ML/solution.txt", psol, fmt='%1.2f', newline='')

# 4
ages = data['Age'].dropna()
avr = ages.mean()
med = ages.median()
med_and_avr = np.array([avr, med])
np.savetxt("C:/Python/ML/solution.txt", med_and_avr, fmt='%4.1f', newline='')

# 5
corr = data['SibSp'].corr(data['Parch'])
med_and_avr = np.array([corr])
np.savetxt("C:/Python/ML/solution.txt", med_and_avr, fmt='%1.2f', newline='')


# 6
def clean_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)

    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)

    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')

    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
sol_name = np.array([name_counts.head(1).index.values[0]])
sol_name.tofile("C:/Python/ML/solution.txt", sep=" ")