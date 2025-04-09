#!/usr/bin/env python
# coding: utf-8

# - STEP 1: IMPORTING LIBRARIES

# In[47]:


pip install xgboost


# In[168]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import warnings
warnings.filterwarnings("ignore")


# - STEP 2: LOADING THE DATA

# In[169]:


df=pd.read_csv("cars_engage_2022.csv")
df


# - STEP 3: BASIC UNDERSTANDING OF DATA

# how big is the data

# In[53]:


df.shape


# In[57]:


pd.set_option


# In[62]:


pd.set_option("display.max_rows",None)


# - PREVIEW

# In[63]:


df.sample(2)


# In[64]:


df.drop(columns="Unnamed: 0",inplace=True)


# In[187]:


df.shape


# - FETCHING COLUMNS NAMES AND TYPES

# In[188]:


df.dtypes


# - BASIC INFORMATION OF DATA

# In[189]:


df.info()


# -  from above output we can observe that mjority of the columns are object types  i.e, 134 and only 6 columns are of numeric type

# - STEP: PREPROCESSING  

# In[190]:


df.isnull().sum()


# In[191]:


ms_count=[]
ms_count_per=[]
col=[]
dtypes=[]
for i in df.columns:
    if df[i].isnull().sum()>0:
        ms_count.append(df[i].isnull().sum())
        ms_count_per.append(df[i].isnull().sum()*100/df.shape[0])
        col.append(i)
        dtypes.append(df[i].dtypes)

ms=pd.DataFrame({"col":col,"dtypes":dtypes,"missing_count":ms_count,"missing_count_per":ms_count_per})
ms
    


# In[193]:


len(ms)


# In[195]:


ms_count=[]
ms_count_per=[]
col=[]
dtypes=[]
for i in df.columns:
    if df[i].isnull().sum()*100/df.shape[0]>70:
        ms_count.append(df[i].isnull().sum())
        ms_count_per.append(df[i].isnull().sum()*100/df.shape[0])
        col.append(i)
        dtypes.append(df[i].dtypes)

ms_70=pd.DataFrame({"col":col,"dtypes":dtypes,"missing_count":ms_count,"missing_count_per":ms_count_per})
ms_70
    


# In[196]:


len(ms_70)


# In[197]:


for i in df.columns:
    if (df[i].isnull().sum()*100/df.shape[0])>=70:
        df.drop(columns=[i],inplace=True)
        
        
        


# In[198]:


df.shape


# In[199]:


ms_count=[]
ms_count_per=[]
ms_dtype=[]
col=[]
for i in df.columns:
    if df[i].isnull().sum()>0:
        ms_count.append(df[i].isnull().sum())
        ms_count_per.append(df[i].isnull().sum()*100/df.shape[0])
        ms_dtype.append(df[i].dtypes)
        col.append(i)
ms=pd.DataFrame({"col":col,"dtypes":ms_dtype,"missing_count":ms_count,"missing_count_per":ms_count_per})
ms                   


# In[200]:


len(ms)


# since the column is 109 where it has missing values so to handle them conviently we seggreate the data in numeric and object

# ### DATA CLEANING

# In[230]:


df_obj=df.select_dtypes(include='object')
df_obj.head(3)


# In[203]:


len(df_obj.columns)


# In[204]:


df_numeric=df.select_dtypes(exclude='object')
df_numeric.head(3)


# In[207]:


len(df_numeric.columns)


# from 115 columns there are :6 are of numeric based and 110 are of object based sowe handle them individual

# In[208]:


ms_count=[]
ms_count_per=[]
ms_dtype=[]
col=[]
for i in df_numeric.columns:
    if df_numeric[i].isnull().sum()>0:
        ms_count.append(df_numeric[i].isnull().sum())
        ms_count_per.append(df_numeric[i].isnull().sum()*100/df.shape[0])
        ms_dtype.append(df_numeric[i].dtypes)
        col.append(i)
ms_numeric=pd.DataFrame({'columns':col,'missing_values':ms_count,'missing_values_%':ms_count_per,'columns_dtypes':ms_dtype})
ms_numeric


# since missing value % is less so we fill them with their corresponding median since median is robust to outlies 

# In[209]:


for  i in df_numeric.columns:
    if df_numeric[i].isnull().sum()>0:
        df_numeric[i]=df_numeric[i].fillna(df_numeric[i].median())


# In[210]:


df_numeric.isnull().sum().sum()


# In[211]:


len(df_obj.columns)


# In[212]:


df_obj.head(2)


# displaying the preview of some data of categorical columns

# In[213]:


df_obj.sample(5)


# In[ ]:


how many columns are of categorical types


# In[215]:


df_obj.shape


# there are many numerical feature which are having object data type but it should be numerical data type

# #### NOTE:

# THERE ARE 110 CATEORICAL FEATURE BUT ALL THEM ARE NOT RELEVANT FOR PREDICTING PRICE OF CAR. SO WE WILL EXTRACT THE FEATURE WHICH ARE RELEVANT TO OUR TARGET FEATURE

# ### Extracting the relevant attributes from the categorical data_frame

# In[228]:


df_obj.head().columns


# In[218]:


df_obj_list = ['make','model','variant','ex-showroom_price','displacement','emission_norm','engine_location','fuel_tank_capacity','heigth','length','weidth','body_type','ARAI_certificates','kerb_weigth','ground_clearence','boot_space','power_steering','keyless_entry','power','torque','odometer','speedometer','techometer','tripmeter','seats_material','fuel_type','wheelbase','central_locking','child_safety_locks','low_fuel_warning','third_row_ac_vents','second_row_ac_vvents','auto_dimming_rear_view_mirro','engine_immobilizer','ABS','EBD','cooled_glove_box','EBA','gear_shift_reminder','adjustable_steering_column','parking_assistance','key_off_reminder','USB_compatibility','bluetooth','seat_heigth_adjustable','navigation_system','turbocharger','automatic_headlamps','cruise_control','welcome_light','USB_port','electric_range']


# In[531]:


df_obj_list


# In[532]:


len(df_obj_list)


# In[533]:


df_obj = df_obj.loc[:, df_obj_list]
df_obj


# In[238]:


len(df_obj.columns)


# In[223]:


df_obj.head(2)


# computing the missing values of categorical dataframe

# In[224]:


ob=[]
ob_nullvalues=[]
ob_nullvalues_per=[]
for i in df_obj.columns:
    if df_obj[i].isnull().sum()>0:
        ob.append(i)
        ob_nullvalues.append(df_obj[i].isnull().sum())
        ob_nullvalues_per.append(df_obj[i].isnull().sum()*100/df.shape[0])
ob_nulldf=pd.DataFrame({'categorical_columns':ob,'categorical_nullvalues':ob_nullvalues,'categorical_nullvalues_%':ob_nullvalues_per})  
ob_nulldf                                 


# In[226]:


len(ob_nulldf)


# In[240]:


df_obj['Make'].mode()


# Now, we would be doing the handling missing values process of above columns one by one and also clean all categorical columns

# In[243]:


df_obj[df_obj['Make'].isnull()]


# In[244]:


df_obj['Make'].isnull().sum()


# In[259]:


df[df['Make'].isnull()]


# In[258]:


df[df['Make'].isnull()]['Model'].unique()


# In[250]:


df_obj['Make'].nunique()


# From above data we can see that mercedes,go+ and rolls royce cars whose company name is missing values so fill them accordingly
# 
# 
# - so we can fill the following car manufacture
# 
# -1 Mercedes-Benz
# -2 Rolls-Royce
# -3 Datsun

# In[260]:


df_obj['Make'] = df_obj['Make'].combine_first(df_obj['Model'])


# In[ ]:





# In[261]:


df_obj['Make'].unique()


# In[262]:


df_obj['Make'].isnull().sum()


# In[264]:


for i in df_obj['Make']:
    if 'Mercedes' in i:
        df_obj['Make']=df_obj['Make'].replace(i,'Mercedes-Benz')
    elif 'Rolls' in i:
        df_obj['Make']=df_obj['Make'].replace(i,'Rolls-Royce')
    elif ' go' in i:
        df_obj['Make']=df_obj['Make'].replace(i,'Datsun')


# In[254]:


df[df['Make']=='Land Rover Rover']


# #### Taking make column since make indicates the car company name so we can not fill with aggregated method moe since it violate the data so we are fetching the rows where make has missing vlues in order to see corresponding car name to get car manufacture

# In[265]:


df_obj['Make'].unique()


# In[266]:


df_obj.columns


# ### taking model

# In[267]:


df_obj['Model'].isnull().sum()


# In[245]:


df_obj['Model'].unique()


# ### taking varient

# In[270]:


df_obj['Variant'].isnull().sum()


# ### taking ex show-room price

# In[284]:


df_obj['Ex-Showroom_Price'].isnull().sum()


# In[294]:


df_obj['Ex-Showroom_Price'].unique()


# In[534]:


df_obj['Ex-Showroom_Price']=df_obj['Ex-Showroom_Price'].str.replace('Rs. ',"").str.replace(",","")
df_obj['Ex-Showroom_Price'].unique()


# In[302]:


df_obj['Ex-Showroom_Price']=df_obj['Ex-Showroom_Price'].astype(float)


# In[303]:


df_obj['Ex-Showroom_Price'].dtypes


# In[298]:


df.columns


# ### taking displacement columns

# In[304]:


df_obj['Displacement'].unique()


# Removing "CC" value from "Displacement" attribute

# In[306]:


df_obj['Displacement']=df_obj['Displacement'].str.replace("cc","")


# In[307]:


df_obj['Displacement'].unique()


# ### converting data type of displacement attribute

# In[308]:


df_obj['Displacement']=df_obj['Displacement'].astype(float)


# In[309]:


df_obj['Displacement'].dtypes


# ## detecting the missing values

# In[314]:


df_obj['Displacement'].isnull().sum()*100/df.shape[0]


# In[313]:


df_obj['Displacement'].median()


# In[312]:


df_obj[df_obj['Displacement'].isnull()]


# In[321]:


df_obj['Displacement'].isnull().sum()


# ## taking "power" column

# In[318]:


df_obj['Power'].unique()


# #### computing the missing value in Power columns

# In[322]:


print('Total missing value is',df_obj['Power'].isnull().sum())


# - There are no missing values presnet in Power feature
# - From the unique values we can observe different unit of measure for power values
# - So its important to convert all units to a single unit because machine learning models require uniformity in the data
# - Hence we can convert all the power values to a single unit which can be HP 
# - 1 PS is equivalent to 0.98632 HP
# - 1 KW is equivalent to approximately 1.34 HP 
# - Some of the value are having Nm unit of measurement  but in reality they are units fo torque values
# 

# In[324]:


df_obj[(df_obj['Power']=="1600Nm@2000-6000rpm")]


# In[325]:


df_obj.loc[356,'Power']="1479bhp@6700rpm"
df_obj.loc[356,'Torque']="1600Nm@2000-6000rpm"


# In[329]:


new_power=[]          # to store all the converted power values

for i in df_obj['Power']:
    power_value=i.split("@")[0].replace(" ","").lower()
    
    if "ps" in power_value:
        value1 = round((float(power_value.replace(" ps ",""))*0.98632),2)
        new_power.append(value1)
        
    elif "bhp" in power_value:
        value2 = round((float(power_value.replace(" bhp",""))*1.01387),2)
        new_power.append(value2)
    
    elif "hp" in power_value:
        value3 =float(power_value.replace(" hp",""))
        new_power.append(value3)
     
    elif "kw" in power_value:
        value4 = round((float(power_value.replace(" kw",""))*1.34),2)
        new_power.append(value4)
    
    else:
        new_power.append(power_value)


# In[ ]:





# ### Taking " Torque " Column

# In[332]:


df_obj['Torque'].isnull().sum()


# In[333]:


df_obj[df_obj['Torque'].isnull()]


# In[335]:


df_obj['Torque'].unique()


# In[414]:


df_obj[df_obj['Torque'].isnull()][["Make","Model","Variant","Torque"]]


# In[ ]:





# In[338]:


df_obj.loc[536,"Torque"]= "195Nm@1400RPM"
df_obj.loc[1158,"Torque"]= "400Nm@1500-4400RPM"


# In[339]:


df_obj['Torque'].isnull().sum()


# In[340]:


df_obj['Torque'].unique()


# - From the unique values we can observe two different unit of measure for torque values
# - So its important to convert both the units to a single unit because machine learning models require uniformly in the data
# - So we can converted all the torque vlaue to a single  unit which can be used in machine learning model Nm(Newton meter)
# 

# In[342]:


new_torque = []
for i in df_obj['Torque']:
    torque_value = i.split("@")[0].replace(" ","").lower()
    print(torque_value)


# In[343]:


new_torque = []

for i in df_obj['Torque']:\
    torque_value = i.split("@")[0].replace(" ","").lower()
    
    if "nm" in torque_value:
        value1 = float(torque_value.replace("nm",""))
        new_torque.append(value1)
    elif "kgm" in torque_value:
        value2 =round(float(torque_value.replace("kgm","")*9.80665),2)
        new_torque.append(value2)
    else:
        new_torque.append(torque_value)


# In[344]:


df_obj['Torque'].unique()


# In[ ]:





# ### Taking " Mileage " Column

# In[348]:


df_obj['ARAI_Certified_Mileage'].isnull().sum()*100/df_obj.shape[0]


# In[349]:


df_obj['ARAI_Certified_Mileage'].unique()


# In[350]:


df_obj['ARAI_Certified_Mileage']=df_obj['ARAI_Certified_Mileage'].str.replace('km/litre',"")


# In[351]:


df_obj['ARAI_Certified_Mileage'].unique()


# In[352]:


df_obj[df_obj['ARAI_Certified_Mileage']=='9.8-10.0']


# In[353]:


df_obj.loc[304,'ARAI_Certified_Mileage']=9.8


# In[354]:


df_obj.loc[304]


# In[356]:


df_obj[df_obj['ARAI_Certified_Mileage']=="10Kmpl"]


# In[357]:


df_obj.loc[851,'ARAI_Certified_Mileage']=10
df_obj.loc[851]


# In[358]:


df_obj[df_obj['ARAI_Certified_Mileage']=='22.4-21.9']


# In[359]:


df_obj.loc[353,"ARAI_Certified_Mileage"]=22.4
df_obj.loc[353]


# In[360]:


df_obj['ARAI_Certified_Mileage'].unique()


# In[362]:


df_obj[df_obj['ARAI_Certified_Mileage']=='1449']


# In[363]:


df_obj[df_obj['ARAI_Certified_Mileage']=='142']


# In[364]:


df_obj.loc[1036,"ARAI_Certified_Mileage"]=16.9
df_obj.loc[[794,795,799,800,],"ARAI_Certified_Mileage"]=16.1
df_obj['ARAI_Certified_Mileage'].unique()


# ##### computing the missing values % in ARAI_Certified_Mileage column

# In[370]:


print("Total missing values is",df_obj['ARAI_Certified_Mileage'].isnull().sum()*100/df.shape[0])


# In[371]:


df_obj[df_obj['ARAI_Certified_Mileage'].isnull()]


# In[412]:


median_value = df_obj.groupby(["Make"])['ARAI_Certified_Mileage'].median()
median_value
index = df_obj[df_obj['ARAI_Certified_Mileage'].isnull()].index
index


# In[ ]:





# ### Taking " Ground_Clearence" Column

# In[380]:


df_obj['Ground_Clearance'].isnull().sum()


# In[381]:


df_obj['Ground_Clearance'].unique()


# ##### computing the missing value of ground clearence

# In[382]:


print(" missing value % in ground clearence is ",df_obj['Ground_Clearance'].isnull().sum()*100/df.shape[0])


# - so we filll them sccordingly on their respective car company

# In[384]:


df_obj[df_obj['Ground_Clearance'].isnull()]


# In[ ]:





# ### Taking Boot_Space Column

# In[385]:


df_obj['Boot_Space'].isnull().sum()


# In[386]:


df_obj['Boot_Space'].unique()


# In[387]:


type(np.nan)


# In[389]:


index= 0

for i in df_obj["Boot_Space"]:
    if type(i) == float:
        df_obj.loc[index,"Boot_Space"]=i
    else:
        df_obj.loc[index,"Boot_Space"]=i[0:4].replace(" ","")
    index+=1    


# In[390]:


df_obj['Boot_Space'].unique()


# In[392]:


df_obj['Boot_Space']=df_obj['Boot_Space'].str.replace("l","").str.replace(" ","")


# In[393]:


df_obj['Boot_Space'].unique()


# #####  computing the missing value % in Boot_Space

# In[394]:


print(" the missing value % in Boot_Space is ",df_obj['Boot_Space'].isnull().sum()*100/df.shape[0])


# In[397]:


l =  ['Height','Length','Width','Wheelbase']

for i in l:
    print(i,"................",df_obj[i].unique().tolist(),"\n\n")


# ### observation 
# 
# - All the feature are having same unit of measure which is mm. Hence we can simply extract numerical value  and convert its data type

# In[398]:


for col in l:
    df_obj[col]=df_obj[col].apply(lambda x: str(x).replace(" mm","")).astype(float)


# In[399]:


l=['Height','Length','Width','Wheelbase']

for i in l:
    print(i,"................",df_obj[i].unique().tolist(),"\n\n")


# #### computing the missing value in Height ,Length,Width,Wheelbase
# 

# In[400]:


df_obj[l].isnull().sum()


# first find the data where heigth has missing value

# In[401]:


df_obj[df_obj['Height'].isnull()]


# In[403]:


df_obj.loc[314,"Height"]=1387


# In[404]:


df_obj['Height'].isnull().sum()


# find the data where width has missing value

# In[405]:


df_obj[df_obj['Width'].isnull()]


# In[406]:


df_obj.loc[314,'Width']=1866


# In[407]:


df_obj['Width'].isnull().sum()


# In[408]:


df_obj.loc[440:445,"Width"]=1645


# In[409]:


df_obj['Width'].isnull().sum()


# find the data where Wheelbase has missing values

# In[410]:


df_obj[df_obj['Wheelbase'].isnull()]


# In[422]:


wheelbase_median = df_obj.groupby(['Make'])['Wheelbase'].median().tolist()
index_list = df_obj[df_obj['Wheelbase'].isnull()].index

for index in index_list:
    manufacturer = df_obj['Make'][index]
    value = wheelbase_median['Wheelbase'][manufacturer]
    df_obj.loc[index,'Wheelbase'] = value


# In[418]:


df_obj['Wheelbase'].isnull().sum()


# In[ ]:


#####################################################################


# ### Taking " Fuel Tank" Column

# In[423]:


df_obj['Fuel_Tank_Capacity'].unique()


# ### observation
# 
# - All the values of fuel tank capacity are in same unit of measure which is litre. Hence we can simply extract numerical values and convert its data type

# In[426]:


df_obj['Fuel_Tank_Capacity']=df_obj['Fuel_Tank_Capacity'].apply(lambda x: x.split(" ")[0] if type(x)==str else x)
df_obj['Fuel_Tank_Capacity'].unique()


# In[427]:


df_obj['Fuel_Tank_Capacity']=df_obj['Fuel_Tank_Capacity'].astype(float)


# ##### computing the missing values in fuel tank capacity

# In[428]:


df_obj['Fuel_Tank_Capacity'].isnull().sum()


# In[429]:


df_obj[df_obj['Fuel_Tank_Capacity'].isnull()]


# In[430]:


df_obj['Fuel_Tank_Capacity']=df_obj['Fuel_Tank_Capacity'].fillna(df_obj['Fuel_Tank_Capacity'].median())


# In[431]:


df_obj['Fuel_Tank_Capacity'].isnull().sum()


# In[ ]:


####################################################


# ### Taking " Kerb_Weight" Column

# In[432]:


df_obj['Kerb_Weight'].unique()


# In[435]:


df_obj['Kerb_Weight']=df_obj['Kerb_Weight'].apply(lambda x: x.split(" - ")[0] if type(x)==str else x)


# In[436]:


df_obj['Kerb_Weight'].unique()


# ##### computing the missing value of kerb_weight

# In[439]:


df_obj['Kerb_Weight'].isnull().sum()


# In[441]:


df_obj[df_obj['Kerb_Weight'].isnull()]


# filling the missing value in kerb_weight corresponding to car manufacturer according as follow

# In[442]:


median = df_obj.groupby(['Make'])['Kerb_Weight'].median()
index = df_obj[df_obj['Kerb_Weight'].isnull()].index

for i in index:
    manufacturer = df_obj[df_obj['Kerb_Weight'].isnull()]['Make'][i]
    value = median['Kerb_Weight'][manufacturer]
    df_obj.loc[i,'Kerb_Weight'] = value


# In[443]:


df_obj['Kerb_Weight'].isnull().sum()


# In[444]:


manufacturer = df_obj[df_obj['Kerb_Weight'].isnull()]
len(manufacturer)


# In[445]:


index df_obj[df_obj['Kerb_Weight'].isnull()].index

for i in index:
    manufacturer = df_obj[df_obj['Kerb_Weight'].isnull()]['Make'][i]
    if manufacturer ==' Bajaj':
        df_obj.loc[i,'Kerb_Weight'] = 451
    elif manufacturer == 'Force':
        df_obj.loc[i,'Kerb_Weight'] = 2050
    elif manufacturer == 'Kia':
        df_obj.loc[i,'Kerb_Weight'] = 2195
    elif manufacturer == 'Mg':
        df_obj.loc[i,'Kerb_Weight'] = 1613


# In[ ]:


#####################################################################


# ### Taking " Odometer","Speedometer","Tachometer","Tripmeter" Columns

# 

# In[447]:


l = ['Odometer','Speedometer','Tachometer','Tripmeter']

for i in l:
    print(i,"..............",df_obj[i].unique())


# ##### computing the missing value

# In[448]:


df_obj[l].isnull().sum()


# In[449]:


df_obj[l].mode()


# ### Since we fill NAN with yes in odometer , speedometer, as most of the cars is having odometer and speedometer and fill nan in tachometer and tripmeter with not_defined

# In[450]:


df_obj['Odometer']=df_obj['Odometer'].fillna('Yes')
df_obj['Speedometer']=df_obj['Speedometer'].fillna('Yes')
df_obj['Tachometer']=df_obj['Tachometer'].fillna('Not defined')
df_obj['Tripmeter']=df_obj['Tripmeter'].fillna('Not defined')


# In[452]:


df_obj[['Odometer','Speedometer','Tachometer','Tripmeter']].isnull().sum()


# In[ ]:


##########################################################################


# ### Taking " Drivetrain" Column

# In[453]:


df_obj['Drivetrain'].unique()


# since drivetrain is having ambiguity AWD with 4WD and we are correcting as follow and making other as AWD , FWD, RWD

# In[456]:


df_obj['Drivetrain']=df_obj['Drivetrain'].replace("FWD ( front wheel drive)","FWD")
df_obj['Drivetrain']=df_obj['Drivetrain'].replace("RWD ( Rear wheel drive)","RWD")
df_obj['Drivetrain']=df_obj['Drivetrain'].replace("AWD ( All wheel drive)","AWD")
df_obj['Drivetrain']=df_obj['Drivetrain'].replace("4WD ","AWD")


# In[457]:


df_obj['Drivetrain'].unique()


# computing the data where Drivetrain has missing value

# In[458]:


df_obj[df_obj['Drivetrain'].isnull()]


# In[463]:


df_obj['Drivetrain'].isnull().sum()


# In[465]:


index_list = df_obj[df_obj['Drivetrain'].isnull()].index

for index in index_list:
    
        if df_obj['Make'][index] in ['Mini','Maserati','Jaguar']:
            df_obj.loc[index,'Drivetrain'] = 'AWD'
        elif df_obj['Make'][index] =='Mercedes-Benz':
            df_obj.loc[index,'Drivetrain'] = 'RWD'
        elif df_obj['Make'][index] =='Tata':
            df_obj.loc[index,'Drivetrain'] = 'FWD'
                


# In[466]:


df_obj['Drivetrain'].isnull().sum()


# In[ ]:


####################################################################


# ### Taking " Emission_norm " Columns

# In[467]:


df_obj['Emission_Norm'].unique()


# ### from above output we can say that there is ambiguity with BS 6 with BS VI

# In[468]:


df_obj['Emission_Norm']=df_obj['Emission_Norm'].str.replace("BS 6 "," BS VI")


# In[469]:


df_obj['Emission_Norm'].unique()


# In[470]:


df_obj['Emission_Norm'].isnull().sum()


# In[471]:


df_obj[df_obj['Emission_Norm'].isnull()]


# #### fill the missing value according  to corresponding car manufacturer

# In[472]:


index_list = df_obj[df_obj['Emission_Norm'].isnull()].index

for index in index_list:
    
    if (df_obj['Make'][index]=='Mahindra') & (df_obj['Model'][index]=='Alturas G4'):
        df_obj.loc[index,'Emission_Norm'] = ' BS VI'
        
    elif (df_obj['Make'][index]=='Hyundai') & (df_obj['Model'][index]=='Kona Electric'):
          df_obj.loc[index,'Emission_Norm'] = ' No Emissions'
        
    elif (df_obj['Make'][index]) in ['Honda','Aston Martin','Land Rover','Jaguar']:
         df_obj.loc[index,'Emission_Norm'] = ' BS VI'
     
    elif (df_obj['Make'][index]=='Mahindra') & (df_obj['Model'][index]=='E Verito'):
         df_obj.loc[index,'Emission_Norm'] = ' No Emissions'
            
    elif (df_obj['Make'][index]=='Mitsubishi') & (df_obj['Model'][index]=='Outlander'):
          df_obj.loc[index,'Emission_Norm'] = ' BS VI'


# In[473]:


df_obj['Emission_Norm'].isnull().sum()


# In[ ]:


##########################################################################


# ### Taking " Cylinder_configuration "," Power_steering "," Keyless_entry"

# In[474]:


cols = ['Cylinder_Configuration','Power_Steering','Keyless_Entry']

for i in cols:
    print(i,"............",df_obj[i].unique())
    
    


# ##### computing the missing values of cylinder configuration

# In[476]:


df_obj['Cylinder_Configuration'].isnull().sum()


# In[477]:


df_obj[df_obj['Cylinder_Configuration'].isnull()]


# we fill the NAN in cylinder configuration with not defined as we can't fill with mode or and random value because it may violate the data

# In[478]:


df_obj['Cylinder_Configuration']=df_obj['Cylinder_Configuration'].fillna('not defined')


# In[479]:


df_obj['Cylinder_Configuration'].isnull().sum()


# ### Cleaning Power_Steering

# In[480]:


df_obj['Power_Steering'].unique()


# from above there is ambiguity in electric power, hydraulic power with electro hydraulic so we are correcting as below

# In[483]:


df_obj['Power_Steering']=df_obj['Power_Steering'].replace('Electric Power ','Hydraulic Power ','Electro-Hydraulic')
df_obj['Power_Steering']


# In[484]:


df_obj['Power_Steering']=df_obj['Power_Steering'].fillna('not defined')


# In[485]:


df_obj['Power_Steering'].isnull().sum()


# In[ ]:


###########


# #### Cleaning Keyless Entry

# In[487]:


df_obj['Keyless_Entry'].unique()


# above output we can sat there is ambigutiy in remote,smart key and smart key remote

# In[488]:


df_obj['Keyless_Entry']=df_obj['Keyless_Entry'].replace('Remote, Smart Key','Smart Key Remote')


# In[489]:


df_obj['Keyless_Entry'].unique()


# In[490]:


df_obj['Keyless_Entry'].isnull().sum()


# we can't fill the random values in this so wwe fill with not defined so that integrity of data won't get affected

# In[491]:


df_obj['Keyless_Entry']=df_obj['Keyless_Entry'].fillna('Not defined')


# In[492]:


df_obj['Keyless_Entry'].isnull().sum()


# In[ ]:


##############


# ### Cleaning Adjustable_Steering

# In[494]:


df_obj['Adjustable_Steering_Column'].unique()


# In[495]:


df_obj['Adjustable_Steering_Column']=df_obj['Adjustable_Steering_Column'].replace('Rake, Reach','Reach, Rake')


# In[496]:


df_obj['Adjustable_Steering_Column'].unique()


# In[497]:


df_obj['Adjustable_Steering_Column'].isnull().sum()


# since we can't fill the random values in this so we fill with not defined so that integrity of data wont get affect

# In[498]:


df_obj['Adjustable_Steering_Column']=df_obj['Adjustable_Steering_Column'].fillna('Not defined')


# In[499]:


df_obj['Adjustable_Steering_Column'].isnull().sum()


# In[ ]:


############


# ### Handling Body Type

# In[500]:


df_obj['Body_Type'].unique()


# from above output we can see there is ambiguity in crossover , SUV == SUV, corssover so we are correcting as :

# In[501]:


df_obj['Body_Type']=df_obj['Body_Type'].replace('SUV, Crossover','SUV')


# In[502]:


df_obj['Body_Type'].unique()


# ##### computing the missing value in body type

# In[505]:


df_obj['Body_Type'].isnull().sum()


# we can not fill with mode so we fill with not deifned

# In[506]:


df_obj['Body_Type']=df_obj['Body_Type'].fillna('Not defined')


# In[507]:


df_obj['Body_Type'].isnull().sum()


# ### Taking Parking_Assitance

# In[508]:


df_obj['Parking_Assistance'].unique()


# from above output we can say there is ambiguity as below

# In[509]:


df_obj['Parking_Assistance'].replace({"Rear sensors, Rear sensors with camera":"Rear sensors with camera",})


# In[510]:


df_obj['Parking_Assistance'].nunique()


# In[511]:


df_obj['Parking_Assistance'].isnull().sum()


# filling the missing value in parking assistance with not defined

# In[512]:


df_obj['Parking_Assistance']= df_obj['Parking_Assistance'].fillna('Not defined')


# In[513]:


df_obj['Parking_Assistance'].isnull().sum()


# In[ ]:


#################################


# ### Taking Engine Location 

# In[514]:


df_obj['Engine_Location'].unique()


# In[515]:


df_obj['Engine_Location'].isnull().sum()


# - filling the missing value in engine location witth not defined

# In[516]:


df_obj['Engine_Location']=df_obj['Engine_Location'].fillna('Not defined')


# In[ ]:


####################


# ### Taking Seat_Material

# In[517]:


df_obj['Seats_Material'].unique()


# computing the missing values in seat material 

# In[518]:


df_obj['Seats_Material'].isnull().sum()


# - filling the missing valuein seat material with not defined

# In[519]:


df_obj['Seats_Material']=df_obj['Seats_Material'].fillna('Not defined')


# In[520]:


df_obj['Seats_Material'].isnull().sum()


# In[ ]:


##################


# ### Taking tthird_row_ac_vents

# In[522]:


df_obj['Third_Row_AC_Vents'].unique()


# In[523]:


df_obj['Third_Row_AC_Vents'].isnull().sum()


# In[524]:


df_obj['Third_Row_AC_Vents']=df_obj['Third_Row_AC_Vents'].fillna('Not defined')


# In[525]:


df_obj['Third_Row_AC_Vents'].unique()


# In[544]:


cols = ['Central_Locking','Child_Safety_Locks','Low_Fuel_Warning','Second_Row_AC_Vents','Auto-Dimming_Rear-View_Mirror','Engine_Immobilizer','ABS_(Anti-Lock_Braking_System)','EBD_(Electronic_Brake-force_Distribution)','Cooled_Glove_Box','EBA_(Electronic_Brake_Assist)','ESP_(Electronic_Stability_Program)','Gear_Shift_Reminder','Key_Off_Reminder','USB_Compatibility','Bluetooth','Navigation_System','Turbocharger','Automatic_Headlamps']


# In[545]:


cols


# In[546]:


for column in cols:
    print(f"Unique values in {column} is" , df_obj[column].unique().tolist())
    print("-"*100)


# In[547]:


df_obj[cols].isnull().sum()


# In[548]:


df_obj[cols] = df_obj[cols].fillna('Not defined')


# ## Concatinating the numeric and object type data  frame

# In[549]:


df = pd.concat([df_numeric,df_obj],axis=1)


# - checking the shape again

# In[550]:


df.shape


#  ### STEP 5: EXPLORATORY DATA ANALYSIS AND INSIGHTS

# preview of data

# In[552]:


df.head(2)


# In[554]:


df.rename(columns={'Ex-Showroom_Price':'Price'},inplace=True)


# In[555]:


df.sample(2)


# #### since price is main columns so we do the EDA from price pov

#  ##  univariate analysisof price

# In[568]:


plt.figure(figsize=(10,4),facecolor="pink")
plt.subplot(1,2,1)
sns.distplot(df["Price"])
plt.title('distribution of price')
plt.subplot(1,2,1,)
sns.boxplot(['Price'])
plt.tile(' outliers in price')


# #### from above output we can observethe distribution of price is right skewness means there is outliers mean there are some expensive car in the data 

# ### there are highly extreme outliers indicate the extreme highly expensive cars

# ###### detail of expensive cars

# In[572]:


df[df['Price']>16000000]


# #### find the min,max.mean price of the cars in the catelog

# In[562]:


df['Price'].agg(['max','min','mean'])


# #####  detail of cheapest car

# In[573]:


df[df['Price']==df['Price'].min()]


# #### details of expensive cars

# In[574]:


df[df['Price']==df['Price'].max()]


# In[ ]:


### how many cars whose price is greater than mean


# In[576]:


len(df[df['Price']>df['Price'].mean()])*100/df.shape[0]


# In[ ]:


### how many cars whose price is less than mean


# In[577]:


len(df[df['Price']<df['Price'].mean()])*100/df.shape[0]


# ### Maximum,Minimum,and Average price of the each company car in the catalogue

# In[ ]:





# In[578]:


df.groupby(['Make'])['Price'].agg(['min','max','mean'])


# In[581]:


plt.figure(figsize=(12,10))
sns.barplot(x='Make',y='Price',data=df)
plt.xtricks(rotation=90);


# In[583]:


df.corr()["Price"]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




