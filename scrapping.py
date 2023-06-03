#Importing Libraries
import requests
from bs4 import BeautifulSoup
import csv
from house import house_cls
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

#import our finalise model
import pickle

#import models for prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class house_scrape:

    def __init__(self):
        self.house_data =""
        self.fin_house_data =""
       
    
#Importing Data into CSV
    def scrape_houses(self, L_url):
        self.Myhouses = []
#Scraping data
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'}
        self.page= requests.get(L_url, headers = self.headers)
        self.soup = BeautifulSoup(self.page.content,"html.parser")
        self.houses=self.soup.find_all('div',class_='bg-white rounded-4 bg-clip-padding overflow-hidden my-1.2p mx-0.5p tp:border-b-0 shadow-defaultCardShadow tp:shadow-cardShadow tp:mt-0.5p tp:mx-0 tp:mb:1p hover:cursor-pointer nb__2_XSE')
        for house in self.houses:
            self.area=house.find('div',class_='flex flex-col w-33pe items-center tp:w-half po:w-full')            
            self.area_type=self.area.find('div',class_='heading-7').text

            self.location=house.find('h2', class_='heading-6 flex items-center font-semi-bold m-0').a .text
            self.size_div= house.find('div', class_='flex flex-1 pl-0.5p')
            #same class name
            self.size=self.size_div.find('div', class_='font-semibold').text

            self.society=house.find('h2', class_='heading-6 flex items-center font-semi-bold m-0').a .text
            self.sqrft= self.area.find('div', class_='flex').text.replace(",", "")
            self.bath=house.find('div', class_='flex flex-1 border-r border-r-solid border-r-cardbordercolor')
            self.bathroom= self.bath.find('div', class_='font-semibold').text
            self.facing= (house.find('div', class_='font-semibold').text).replace("'","''")
            self.price1= house.find('div', class_='font-semi-bold heading-6')
            self.price_fin = self.price1.find('span').text[1:]
            
        #Inserting data into csv file using getter setter
            
            current_house = house_cls()
            current_house.set_area_type(self.area_type)
            current_house.set_location(self.location)
            current_house.set_size(self.size)
            current_house.set_society(self.society)
            current_house.set_sqrft(self.sqrft)
            current_house.set_bathroom(self.bathroom)
            current_house.set_facing(self.facing)
            current_house.set_price_fin(self.price_fin)
            self.Myhouses.append(current_house)
        return self.Myhouses 
    
#making csv
    def get_pd_table(self, all_houses):
        csv_headers=['area_type','location','size','society','sqrft','bathroom','facing','price_fin']
        with open("House.csv",'w',encoding='utf-8',newline='')as f:
            writer=csv.writer(f)
            writer.writerow(csv_headers)
            for house in all_houses:
                writer.writerow([house.area_type,house.location,house.size,house.society,house.sqrft,house.bathroom,house.facing,house.price_fin])
        #read csv
        self.house_data = pd.read_csv('House.csv')
        return self.house_data

#find median
    def get_median(self):
        #drop some columns to find means and median of bathroom and price
        self.house_data.drop(columns=['area_type','facing','society'],inplace = True )
        
        #sorting lakhs and crores value
        self.new = list(map(lambda x:(denom := x.find('Lacs'), float(x[:denom].strip(" "))*100000 if denom!= -1 else float(x[:x.find('Crores')].strip(" "))*10000000), self.house_data['price_fin']))
        self.new = list(map(lambda x: x[1], self.new))
        print(self.new)
       
        #price value only no.
        self.house_data['price'] = self.new
        print(self.house_data['price'])
        
        #find means and median of bathroom and pric
        median = self.house_data.describe()
        print(median)
        
        #took only 5 column to predict and show
        self.house_data.info()
        self.data = self.house_data.head()
        print(self.data)
    

    def get_values(self):
        #show each valu count seperately
        self.house_data['location'].value_counts ()
        self.house_data['size'].value_counts ()
        
        #remove bhk from bhk
        self.house_data['bhk'] = self.house_data['size'].str.split().str.get(0).astype(int)
        self.house_data['total_sqrft'] = self.house_data['sqrft'].str.split().str.get(0).astype(int)

        #value of sqrft
        self.house_data['total_sqrft'].unique()
        self.data = self.house_data.head()
        print(self.data)


    def get_new_price_mean_col(self):
        #new col 
        self.house_data['price_per_sqrft'] = self.house_data['price'] / self.house_data['total_sqrft']
        print(self.house_data['price_per_sqrft'])
        median = self.house_data.describe()
        print(median)
        
        #drop unneccesary col
        self.house_data.drop(columns=['size','sqrft','price_fin','price_per_sqrft'],inplace = True )
        return self.house_data

    #final data
    def clean_data(self,l_house_data):
        self.Finalhouses = []
        print("printing house data")
        print(l_house_data)

        #extract main location from location
        counter = 0
        for house in range(0, len(l_house_data)):
        
            current_fin_house = house_cls()
            currentdata = l_house_data['location']
            location = ""
            current_loc = currentdata[counter]
            if 'Parel' in current_loc:
                location = 'Parel'
            
            if 'Lower Parel' in current_loc:
                location = 'Parel'

            if 'Lal Baug' in current_loc:
                location = 'Lalbaug'
            
            if 'Chinchpokli' in current_loc:
                location = 'Lalbaug'
            
            if 'Dadar' in current_loc:
                location = 'Dadar'

            if 'Worli' in current_loc:
                location = 'Worli'

            if 'Matunga' in current_loc:
                location = 'Matunga'

            if 'Mahim' in current_loc:
                location = 'Mahim'

            if 'Wadala' in current_loc:
                location = 'Wadala'

            if 'Prabhadevi' in current_loc:
                location = 'Prabhadevi'
            
            #insert final data in csv
            if location != "":
                current_fin_house.set_location(location)#(self.house_data['location'])
                currentdata = l_house_data['total_sqrft']
                current_fin_house.set_total_sqrft(currentdata[counter])
                currentdata = l_house_data['bathroom']
                current_fin_house.set_bathroom(currentdata[counter])#(self.house_data['bathroom'])
                currentdata = l_house_data['bhk']
                current_fin_house.set_bhk(currentdata[counter])#(self.house_data['bhk'])
                currentdata = l_house_data['price']
                current_fin_house.set_price(currentdata[counter])#(self.house_data['price'])
                
                self.Finalhouses.append(current_fin_house)
            counter += 1
        return self.Finalhouses 
    
#making final data csv
    def get_pd_fin_table(self, newhouses):
        csv_headers_fin=['location','total_sqrft','bathroom','bhk','price']
        with open("Final_data.csv",'w',encoding='utf-8',newline='')as f:
            writer=csv.writer(f)
            writer.writerow(csv_headers_fin)
            for house in newhouses:
                writer.writerow([house.location,house.total_sqrft,house.bathroom,house.bhk,house.price])
        #read csv
        self.fin_house_data = pd.read_csv('Final_data.csv')
        if self.fin_house_data.empty == True:
            self.fin_house_data = pd.read_csv('Final_data.csv')
        self.data = self.fin_house_data.head()
        print(self.data)
        
        
        #starting prediction
        self.x =  self.fin_house_data.drop(columns=['price'])
        self.y =  self.fin_house_data['price']
        print(self.y)

        #train test split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)

        print(self.x_train.shape)
        print(self.x_test.shape)

#linear reg
    def linear_reg(self):
        self.col_transformer = make_column_transformer((OneHotEncoder(sparse=False), ['location']), remainder='passthrough')
        self.scaler = StandardScaler()
        self.lr = LinearRegression()
        self.pipe_lr = make_pipeline(self.col_transformer, self.scaler, self.lr)
        print(self.pipe_lr)

        self.pipe_lr.fit(self.x_train, self.y_train)
        self.y_pred_lr = self.pipe_lr.predict(self.x_test)
        self.sqrt_lr = np.sqrt(mean_squared_error(self.y_test, self.y_pred_lr))
        self.score_lr = r2_score(self.y_test, self.y_pred_lr)
        print(self.sqrt_lr, self.score_lr)

#lasso reg
    def lasso_reg(self):
        self.lasso = Lasso()
        self.pipe_ls = make_pipeline(self.col_transformer, self.scaler, self.lasso)
        print(self.pipe_ls)

        self.pipe_ls.fit(self.x_train, self.y_train)
        self.y_pred_lasso = self.pipe_ls.predict(self.x_test)
        self.sqrt_ls = np.sqrt(mean_squared_error(self.y_test, self.y_pred_lasso))
        self.score_ls = r2_score(self.y_test, self.y_pred_lasso)
        print(self.sqrt_ls, self.score_ls)

#ridge reg
    def ridge_reg(self):
        self.ridge = Ridge()
        self.pipe_rd = make_pipeline(self.col_transformer, self.scaler, self.ridge)
        print(self.pipe_rd)

        self.pipe_rd.fit(self.x_train, self.y_train)
        self.y_pred_ridge = self.pipe_rd.predict(self.x_test)
        self.sqrt_rd = np.sqrt(mean_squared_error(self.y_test, self.y_pred_ridge))
        self.score_rd = r2_score(self.y_test, self.y_pred_ridge)
        print(self.sqrt_rd, self.score_rd)

#making final nmodel from highest r2 score
    def final_model(self):
        print("Linear:", self.sqrt_lr, self.score_lr)
        print("Lasso:", self.sqrt_ls, self.score_ls)
        print("Ridge:", self.sqrt_rd, self.score_rd)
        print("Ridge:", self.y_test)
        print(self.y_pred_lr)
        print(self.y_pred_lasso)
        print(self.y_pred_ridge)
        pickle.dump(self.pipe_rd, open('Ridgemodel.pkl', 'wb'))
    

print("!!!!!!!!!!!!Records Inserted Successfully!!!!!!!!!!!!")
print("Done House Data Scraping Successfully")


  










