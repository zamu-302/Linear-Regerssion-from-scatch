import numpy as np
import pandas as pd
import os
def validate_path():
   
    while 1:
      try:
        print("Enter the file name:")
        file=input()
        
        if os.path.exists(file):
            return file
            
        
        
      except:
        print("Make Sure the File Exists in the Same directory")
   

def data_set():
    while 1: 
        print("========Enter the type of data type (1-4) ========")
        print("1.Excel")
        print("2.Csv")
        print("3.Parquet")
        print("4.Json")
    
        try:
            n=int(input())
            if n>4 or n<1:
                print("Please enter a num between 1-4")
                continue
            break
    
        except:
            print("Invalid input! please Enter a number in the range.")
    path=validate_path()
    if n==1:
       try:
        file=pd.read_excel(path)
       except:
          print("Input data Mismatch choice")
          quit()
          
    elif n==2:
        try:
            file=pd.read_csv(path)
        except:
          print("Input data Mismatch choice")
          quit()
    elif n==3:
        try:
            file=pd.read_parquet(path)
        except:
          print("Input data Mismatch choice")
          quit()
    else:
        try:
            file=pd.read_json(path)
        except:
          print("Input data Mismatch choice ")
          quit()
    return file



file= data_set()

def feature(file):
   feat_choice=[]
   for i,col in enumerate(file.columns):
      print(i+1,col)
   while True:
      print("Enter Features to include")
      print("0 to quit")

      n=int(input())
      if n<1 or n>len(file.columns)+1:
         if n==0:
            break
         print("Enter a Choice in Range")
         continue
      
      feat_choice.append(n-1)
   return feat_choice


print("Choose the values to include as features")    
x_feat_choice=feature(file)
while True:
    print("Choose the Value to Predict")
    y_feat_choice=feature(file)
    if len(y_feat_choice)>1:
       print("please Enter only one feature")
       continue
    break

x=file.iloc[:, x_feat_choice].to_numpy(dtype=float)
x = (x - x.mean(axis=0)) / x.std(axis=0)
m = x.shape[0]
X = np.hstack([np.ones((m, 1)), x])
Y=file.iloc[:, y_feat_choice].to_numpy(dtype=float)
Y=(Y-Y.mean())/Y.std()
Theta = np.zeros((X.shape[1], 1)) 



iterations=1000
alpha=0.01
for i in range(iterations):
    y_hat=X @ Theta  #Hypothesis 
    cost=1/(2*m)*(((y_hat-Y).T) @ (y_hat-Y))# Cost function 

    gradient=(1/m)*(X.T @(y_hat-Y))#Graident Function

    Theta=Theta -(alpha*gradient)# Graident decsent

    if i%100==0:
       print(i,cost.item())












            
                
                
            
    
            
    
