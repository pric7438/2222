# imports
from coinbase.wallet.client import Client
import cbpro
import time
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


# connect to coinbase api key and cbpro
# the coinbase api key requires read access to all wallets
# 2222.dat requires the the API Key on line one and API Secret on line two
# the API Key requires the wallet:accounts:read permission for all available wallets
handshake = open('2222.dat', 'r').read().splitlines()
client = Client(handshake[0], handshake[1])
c = cbpro.PublicClient()




##################################
##  BEGIN COLLECT CRYPTO NAMES  ##
##################################


# connect to coinbase api and generate list of crypto names
# only part of the program which accesses the coinbase api key

# initialize next wallet id 
# declare empty list
next = None
names = []

# this loop will run until the next_uri parameter is none
while True:
    
    accounts = client.get_accounts(starting_after = next)
    next = accounts.pagination.next_starting_after
    
    for wallet in accounts.data:
    
        # change crypto name to cbpro historic rates product ticker name
        tempStr = wallet['name']
        tempStr = tempStr.replace(" Wallet", "")
        tempStr = tempStr + "-USD"
        
        # filter out cryptos cbpro can't pull up, i.e. not found and delisted, and stable coins
        # COMMENT THIS OUT EVERY NOW AND AGAIN TO DOUBLE CHECK
        
        if tempStr not in "ETH2-USD" and tempStr not in "REPV2-USD" and tempStr not in "USDC-USD" and tempStr not in "XRP-USD" and tempStr not in "Cash (USD)-USD" and tempStr not in "GNT-USD" and tempStr not in "UST-USD" and tempStr not in "WLUNA-USD" and tempStr not in "MNDE-USD" and tempStr not in "USDT-USD" and tempStr not in "MSOL-USD" and tempStr not in "GUSD-USD":
            
            # add to names list
            names.append(tempStr)
    
    # escape loop            
    if accounts.pagination.next_uri == None:
        break

# OUTPUT FOR TESTING
# print list of names
#for count, string in enumerate(names):
    #print(count + 1, string)
    
    
##################################
###  END COLLECT CRYPTO NAMES  ###
##################################




##################################
####  BEGIN PREDICT FUNCTION  ####
##################################


# this function is called multiple times to predict future crypto prices

# inputs -> algo is the algorithm function name
#           abbrev is the algorithm abbreviation for output
#           scope is the granularity in seconds of the cbpro historic rates

def prediction(algo, abbrev, scope):
    
    # declare empty dictionaries
    result_Avg = {}
    result_Vol = {}
    
    # iterate through each crypto
    for count, string in enumerate(names):
                
        # pull historic rates for crypto: 86400 = 1 day, 3600 = 1 hour, 900 = 15 min, 300 = 5 min, 60 = 1 min
        # get historic rates function returns 300 data points
        # 86400 = 300 days, 3600 = 12.5 days, 900 = 3.125 days, 300 = 1.04 days, 60 = 5 hours
        # this program focuses on short term data
        
        raw = c.get_product_historic_rates(product_id = string, granularity = scope)
        
        # short pause so cbpro calls don't error out
        time.sleep(0.10)
        
        # put in chronological order
        raw.reverse()
        
        # send to pandas dataframe
        df = pd.DataFrame(raw, columns = [ "Date", "Open", "High", "Low", "Close", "Volume" ]) 
                
        # store x and y variables
        # x = df.drop(("Close"), axis = 1)
        # y = df['Close']
        
        # perhaps better way to store x and y variables
        # STICK WITH THIS FOR NOW ... FURTHER TESTING REQUIRED
        x = df.iloc[:, 0:5].values
        y = df.iloc[:, 4].values
        
        # train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)    
    
        # scale the features
        scale = StandardScaler()
        x_train = scale.fit_transform(x_train)
        x_test = scale.transform(x_test)
        
        # pipeline and training
        mlai = make_pipeline(algo)
        mlai.fit(x_train, y_train)
        
        # list of future values
        fv = mlai.predict(x_test)
        
        # next three predicted values
        next3 = fv[:3]
        
        # pull last close price        
        last = df['Close'].iloc[-1]
        
        # calculate average of next three predicted values 
        avg = (( (sum(next3)/len(next3)) - last ) / last) * 100
        avg = round(avg, 2)
    
        # convert to percentages
        next3[0] = float(((next3[0] - last) / last) * 100)    
        next3[1] = float(((next3[1] - last) / last) * 100)
        next3[2] = float(((next3[2] - last) / last) * 100)
        
        # calculate volatility
        vol = abs( max(next3) - min(next3) )
        
        # remove -USD for clean output
        string = string.replace("-USD", "")

        # add to dictionaries
        result_Avg.update( {string : avg} )
        result_Vol.update( {string : vol} )
        
        # OUTPUT FOR TESTING
        # print each crypto name with next three future values and percentage increase/decrease compared to last price
        #fv1 = round((((fv[0] - last) / last) * 100), 2)
        #fv2 = round((((fv[1] - last) / last) * 100), 2)
        #fv3 = round((((fv[2] - last) / last) * 100), 2)


        #print( str(count+1) + ". " + string + " last -> " + str(last) + " future -> " + str(fv[0]) + " " + str(fv[1]) + " " + str(fv[2]) )
        #print("  " + string + " " + str(fv1) + "% " + str(fv2) + "% " + str(fv3) + "%")
        

    # outside of loop    
    sortAvg = dict(sorted(result_Avg.items(), key = operator.itemgetter(1), reverse = True))
    sortVol = dict(sorted(result_Vol.items(), key = operator.itemgetter(1), reverse = True))
    
    # output
    print()
    print("##################################")
    print("            " + abbrev + str(scope) + " TOP 3")
    print("##################################")
    
    print("\nHighest Average Gain")
    print( str(list(sortAvg.keys())[0]) + " -> " + str(sortAvg[list(sortAvg.keys())[0]]) + "%" )
    print( str(list(sortAvg.keys())[1]) + " -> " + str(sortAvg[list(sortAvg.keys())[1]]) + "%" )
    print( str(list(sortAvg.keys())[2]) + " -> " + str(sortAvg[list(sortAvg.keys())[2]]) + "%" )
    
    print("\nMost Volatile")
    print( str(list(sortVol.keys())[0]) + " -> " + str("{:.2f}".format(sortVol[list(sortVol.keys())[0]])) + "%" )
    print( str(list(sortVol.keys())[1]) + " -> " + str("{:.2f}".format(sortVol[list(sortVol.keys())[1]])) + "%" )
    print( str(list(sortVol.keys())[2]) + " -> " + str("{:.2f}".format(sortVol[list(sortVol.keys())[2]])) + "%" )
    print()
 
    
    
    
##################################
#####  END PREDICT FUNCTION  #####
##################################




# void main
prediction(RandomForestRegressor(), "RF", 60)
prediction(RandomForestRegressor(), "RF", 300)
prediction(RandomForestRegressor(), "RF", 900)

prediction(LinearRegression(fit_intercept=True), "LR", 60)
prediction(LinearRegression(fit_intercept=True), "LR", 300)
prediction(LinearRegression(fit_intercept=True), "LR", 900)

prediction(DecisionTreeRegressor(), "DT", 60)
prediction(DecisionTreeRegressor(), "DT", 300)
prediction(DecisionTreeRegressor(), "DT", 900)

prediction(ElasticNet(), "EN", 60)
prediction(ElasticNet(), "EN", 300)
prediction(ElasticNet(), "EN", 900)

prediction(Ridge(), "RD", 60)
prediction(Ridge(), "RD", 300)
prediction(Ridge(), "RD", 900)

prediction(linear_model.Lasso(), "LS", 60)
prediction(linear_model.Lasso(), "LS", 300)
prediction(linear_model.Lasso(), "LS", 900)

prediction(SVR(), "SVR", 60)
prediction(SVR(), "SVR", 300)
prediction(SVR(), "SVR", 900)

prediction(GradientBoostingRegressor(), "GBR", 60)
prediction(GradientBoostingRegressor(), "GBR", 300)
prediction(GradientBoostingRegressor(), "GBR", 900)
