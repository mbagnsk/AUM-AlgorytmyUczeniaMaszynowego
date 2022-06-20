from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as msb
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression


class Application:
    def __init__(self):
        self.credit_data = pd.read_csv('credit_data.csv', index_col=0)
        self.credit_data = self.credit_data.fillna(value="not available")

        self.credit_data.Sex = self.credit_data.Sex.map({'male': 1, 'female': 2})
        self.credit_data.Housing = self.credit_data.Housing.map({'own': 1, 'rent': 2, 'free': 3})
        self.credit_data['Saving accounts'] = self.credit_data['Saving accounts'].map(
            {'not available': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4})
        self.credit_data['Checking account'] = self.credit_data['Checking account'].map(
            {'not available': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4})
        self.credit_data['Purpose'] = self.credit_data['Purpose'].map(
            {'car': 1, 'furniture/equipment': 2, 'radio/TV': 3, 'domestic appliances': 4, 'repairs': 5, 'education': 6,
             'business': 7, 'vacation/others': 8})
        self.credit_data['Risk'] = self.credit_data['Risk'].map(
            {'bad': 0, 'good': 1})

        X = self.credit_data.drop(['Risk'], axis=1)
        y = self.credit_data['Risk']

        self.LogReg = LogisticRegression(penalty='none',
                                    solver='lbfgs',
                                    multi_class='ovr',
                                    class_weight={0: 0.71556351, 1:1.65975104})
        self.LogReg.fit(X, y)

        self.window = tk.Tk()

        self.window.title("Prediction v1")
        self.window.geometry("400x600")

        # Just text - header
        self.label = ttk.Label(text="Credit risk predictior")

        # Sex
        self.label_sex = ttk.Label(text="Please select a sex: ")
        self.sex_value = tk.StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox
        self.sexBox = ttk.Combobox(self.window, textvariable=self.sex_value)  # tworzenie kontrolki Combobox
        self.sexBox['values'] = (
            'male', 'female')  # ustawienie elementów zawartych na liście rozwijanej
        self.sexBox.current(0)  # ustawienie domyślnego indeksu zaznaczenia

        # Job

        self.label_job = ttk.Label(text="Please select a job type: ")
        self.job_value = tk.StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox
        self.jobBox = ttk.Combobox(self.window, textvariable=self.job_value)  # tworzenie kontrolki Combobox
        self.jobBox['values'] = (
            'unskilled and non-resident',
            'unskilled and resident',
            'skilled',
            'highly skilled')  # ustawienie elementów zawartych na liście rozwijanej
        self.jobBox.current(0)  # ustawienie domyślnego indeksu zaznaczenia

        # Housing

        self.label_housing = ttk.Label(text="Please select a housing type: ")
        self.housing_value = tk.StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox
        self.housingBox = ttk.Combobox(self.window, textvariable=self.housing_value)  # tworzenie kontrolki Combobox
        self.housingBox['values'] = (
            'own',
            'rent',
            'free' )  # ustawienie elementów zawartych na liście rozwijanej
        self.housingBox.current(0)  # ustawienie domyślnego indeksu zaznaczenia

        # Savings

        self.label_savings = ttk.Label(text="Please select a savings: ")
        self.savings_value = tk.StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox

        self.savingsBox = ttk.Combobox(self.window, textvariable=self.savings_value)  # tworzenie kontrolki Combobox
        self.savingsBox['values'] = (
            'not available',
            'little',
            'moderate',
            'quite rich',
            'rich')     # ustawienie elementów zawartych na liście rozwijanej
        self.savingsBox.current(0)  # ustawienie domyślnego indeksu zaznaczenia

        # Purpose

        self.label_purpose = ttk.Label(text="Please select a purpose: ")
        self.purpose_value = tk.StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox
        self.purposeBox = ttk.Combobox(self.window, textvariable=self.purpose_value)  # tworzenie kontrolki Combobox
        self.purposeBox['values'] = (
            'car',
            'furniture/equipment',
            'radio/TV',
            'domestic appliances',
            'repairs',
            'education',
            'business',
            'vacation/others')  # ustawienie elementów zawartych na liście rozwijanej
        self.purposeBox.current(0)  # ustawienie domyślnego indeksu zaznaczenia

        # Age

        self.label_age = ttk.Label(self.window, text="Input age: ")
        self.age_value = tk.StringVar()
        self.age_textbox = Entry(self.window, textvariable=self.age_value)

        # Checking amount

        self.label_checking = ttk.Label(text="Please select a checking account: ")
        self.checking_value = tk.StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox
        self.checkingBox = ttk.Combobox(self.window, textvariable=self.checking_value)  # tworzenie kontrolki Combobox
        self.checkingBox['values'] = (
            'not available',
            'little',
            'moderate',
            'quite rich',
            'rich')  # ustawienie elementów zawartych na liście rozwijanej
        self.checkingBox.current(0)  # ustawienie domyślnego indeksu zaznaczenia

        # Credit amount

        self.label_credit = ttk.Label(self.window, text="Input credit amount: ")
        self.credit_value = tk.StringVar()
        self.credit_textbox = Entry(self.window, textvariable=self.credit_value)

        # Duration

        self.label_duration = ttk.Label(self.window, text="Input duration(months): ")
        self.duration_value = tk.StringVar()
        self.duration_textbox = Entry(self.window, textvariable=self.duration_value)

        # submit button

        self.sub_btn = tk.Button(self.window, text='Submit', command=self.submit)

        # grid positions
        self.label.grid(row=0, column=1)

        self.label_age.grid(row=1, column=0)
        self.age_textbox.grid(row=1, column=1)

        self.label_sex.grid(row=2, column=0, pady=(15, 0))
        self.sexBox.grid(row=2, column=1, pady=(15, 0))

        self.label_job.grid(row=3, column=0, pady=(15, 0))
        self.jobBox.grid(row=3, column=1, pady=(15, 0))

        self.label_housing.grid(row=4, column=0, pady=(15, 0))
        self.housingBox.grid(row=4, column=1, pady=(15, 0))

        self.label_savings.grid(row=5, column=0, pady=(15, 0))
        self.savingsBox.grid(row=5, column=1, pady=(15, 0))

        self.label_checking.grid(row=6, column=0, pady=(15, 0))
        self.checkingBox.grid(row=6, column=1, pady=(15, 0))

        self.label_credit.grid(row=7, column=0, pady=(15, 0))
        self.credit_textbox.grid(row=7, column=1, pady=(15, 0))

        self.label_duration.grid(row=8, column=0, pady=(15, 0))
        self.duration_textbox.grid(row=8, column=1, pady=(15, 0))

        self.label_purpose.grid(row=9, column=0, pady=(15, 0))
        self.purposeBox.grid(row=9, column=1, pady=(15, 0))

        self.sub_btn.grid(row=10, column=1)

        self.window.mainloop()

    def submit(self):
        vector = np.empty(9, dtype=object)
        vector[0] = int(self.age_textbox.get())      # age
        vector[1] = self.sexBox.current() + 1        # sex
        vector[2] = self.jobBox.current()            # job
        vector[3] = self.housingBox.current() + 1    # housing
        vector[4] = self.savingsBox.current()        # savings
        vector[5] = self.checkingBox.current()       # checking
        vector[6] = int(self.credit_textbox.get())   # credit amount
        vector[7] = int(self.duration_textbox.get()) # duration
        vector[8] = self.purposeBox.current() + 1    # purpose

        vector = vector.reshape(1, -1)
        y = self.LogReg.predict(vector)

        if y[0] == 0:
            msb.showinfo("Info", 'Prediction: BAD CREDIT RISK')
        else:
            msb.showinfo("Info", 'Prediction: GOOD CREDIT RISK')


apl = Application()