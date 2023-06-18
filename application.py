#! C:/Python30/python

from scrapping import house_scrape
from house import house_cls
import csv
#for gui
from flask import Flask, render_template, request
import pandas as pd
import pickle

house_scr = house_scrape()
#url for data scrapping
url_lalbaug = "https://www.nobroker.in/property/sale/mumbai/Lal%20Baug/?searchParam=W3sibGF0IjoxOC45OTA4MTc3LCJsb24iOjcyLjgzODI1NDcwMDAwMDAxLCJwbGFjZUlkIjoiQ2hJSjVfX0dKZmpPNXpzUkJ2WEpYOF9NQTJnIiwicGxhY2VOYW1lIjoiTGFsIEJhdWciLCJzaG93TWFwIjpmYWxzZX1d&propType=AP&type=BHK1,BHK2,BHK3,BHK4&locality=Lal%20Baug"
url_dadar = "https://www.nobroker.in/property/sale/mumbai/Dadar/?searchParam=W3sibGF0IjoxOS4wMTc3OTg5LCJsb24iOjcyLjg0NzgxMTk5OTk5OTk5LCJwbGFjZUlkIjoiQ2hJSkQ4MmdEdHZPNXpzUjBGdVpPVkJHaWtJIiwicGxhY2VOYW1lIjoiRGFkYXIiLCJzaG93TWFwIjpmYWxzZX1d&propType=AP&type=BHK1,BHK2,BHK3,BHK4&locality=Dadar"
url_matunga = "https://www.nobroker.in/property/sale/mumbai/Matunga/?searchParam=W3sibGF0IjoxOS4wMjY4NzQ3LCJsb24iOjcyLjg1NTMzNTIsInBsYWNlSWQiOiJDaElKejBocjZkRE81enNSSkFNdC0ybHVnWG8iLCJwbGFjZU5hbWUiOiJNYXR1bmdhIiwic2hvd01hcCI6ZmFsc2V9XQ==&propType=AP&type=BHK1,BHK2,BHK3,BHK4&locality=Matunga"
url_mahim = "https://www.nobroker.in/property/sale/mumbai/Mahim/?searchParam=W3sibGF0IjoxOS4wMzUzODQ5LCJsb24iOjcyLjg0MjMwMzYsInBsYWNlSWQiOiJDaElKWFN2WVdpN0o1enNSRGI2RUsyNEoxS1UiLCJwbGFjZU5hbWUiOiJNYWhpbSIsInNob3dNYXAiOmZhbHNlfV0=&propType=AP&type=BHK1,BHK2,BHK3,BHK4&locality=Mahim"
url_wadala = "https://www.nobroker.in/property/sale/mumbai/Wadala%20(W)/?searchParam=W3sibGF0IjoxOS4wMTcxOTY3LCJsb24iOjcyLjg1NzkxNzMsInBsYWNlSWQiOiJDaElKbDc1dDB5UFA1enNSN0hHazBWLUVZMEUiLCJwbGFjZU5hbWUiOiJXYWRhbGEgKFcpIiwic2hvd01hcCI6ZmFsc2V9XQ==&propType=AP&type=BHK1,BHK2,BHK3,BHK4&locality=Wadala%20(W)"
url_prabhadevi = "https://www.nobroker.in/property/sale/mumbai/Prabhadevi/?searchParam=W3sibGF0IjoxOS4wMTYzMjgzLCJsb24iOjcyLjgyOTExMjksInBsYWNlSWQiOiJDaElKY2E3YUxyck81enNSenlXMExmQXpRQWsiLCJwbGFjZU5hbWUiOiJQcmFiaGFkZXZpIiwic2hvd01hcCI6ZmFsc2V9XQ==&propType=AP&type=BHK1,BHK2,BHK3,BHK4&locality=Prabhadevi"

#call function from scrapping
all_houses = []
houses = house_scr.scrape_houses(url_lalbaug)
for house in houses:
    all_houses.append(house)

houses = house_scr.scrape_houses(url_dadar)
for house in houses:
    all_houses.append(house)

houses = house_scr.scrape_houses(url_matunga)
for house in houses:
    all_houses.append(house)

houses = house_scr.scrape_houses(url_mahim)
for house in houses:
    all_houses.append(house)

houses = house_scr.scrape_houses(url_wadala)
for house in houses:
    all_houses.append(house)

houses = house_scr.scrape_houses(url_prabhadevi)
for house in houses:
    all_houses.append(house)

#insert data in csv
house_cls = house_cls()
current_houseData = house_scr.get_pd_table(all_houses)
print('printing value')
for column in current_houseData.columns:
            print(current_houseData[column].value_counts())
current_houseData.isna().sum

#call other functions
house_scr.get_median()
house_scr.get_values()
new_house_data = house_scr.get_new_price_mean_col()
newhouses = house_scr.clean_data(new_house_data)
house_scr.get_pd_fin_table(newhouses)
house_scr.linear_reg()
house_scr.lasso_reg()
house_scr.ridge_reg()
house_scr.final_model()


#GUI
app = Flask(__name__)
data = pd.read_csv('Final_data.csv')
pipe = pickle.load(open("Ridgemodel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    bhks = sorted(data['bhk'].unique())
    bath = sorted(data['bathroom'].unique())
    sqrfts = sorted(data['total_sqrft'].unique())
    print(sqrfts)
    return render_template('index.html', locations=locations, bhks=bhks, bath=bath)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bathroom = request.form.get('bathroom')
    sqrft = request.form.get('total_sqrft')

    print(location, bhk, bathroom, sqrft)

    input = pd.DataFrame([[location,bhk,bathroom,sqrft]], columns=['location', 'bhk', 'bathroom', 'total_sqrft'])
    prediction = pipe.predict(input)[0]
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=80)


 

