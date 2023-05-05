# UK-Accidents_2016_Analysis
 
# EDA and Preprocessing data 

- Load dataset
- Explore the dataset and ask atleast 5 questions to give you a better understanding of the data provided to you. 
- Visualise the answer to these 5 questions.
- Cleaing the data
- Observe missing data and comment on why you believe it is missing(MCAR,MAR or MNAR) 
- Observe duplicate data
- Observe outliers
- After observing outliers,missing data and duplicates, handle any unclean data.
- With every change you are making to the data you need to comment on why you used this technique and how has it affected the data(by both showing the change in the data i.e change in number of rows/columns,change in distrubution, etc and commenting on it).
- Data transformation and feature engineering
- Add a new column named 'Week number' and discretisize the data into weeks according to the dates.Tip: Change the datatype of the date feature to datetime type instead of object.
- Encode any categorical feature(s) and comment on why you used this technique and how the data has changed.
- Identify feature(s) which need normalisation and show your reasoning.Then choose a technique to normalise the feature(s) and comment on why you chose this technique.
- Add atleast two more columns which adds more info to the dataset by evaluating specific feature(s). I.E( Column indicating whether the accident was on a weekend or not). 
- For any imputation with arbitrary values or encoding done, you have to store what the value imputed or encoded represents in a new csv file. I.e if you impute a missing value with -1 or 100 you must have a csv file illustrating what -1 and 100 means. Or for instance, if you encode cities with 1,2,3,4,etc what each number represents must be shown in the new csv file.
- Load the new dataset into a csv file.
- **Extremely Important note** - Your code should be as generic as possible and not hard-coded and be able to work with various datasets. Any hard-coded solutions will be severely penalised.
- Bonus: Load the dataset as a parquet file instead of a csv file(Parquet file is a compressed file format).

# 1 - Extraction

Required Libraries for EDA:


```python
# data manipulation
import pandas as pd
import numpy as np

# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# apply some cool styling
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12,  6)
```

We set an option to see all columns values of the data


```python
pd.set_option('display.max_columns', None)
```

Load the dataset:


```python
df = pd.read_csv("2016_Accidents_UK.csv", low_memory=False)
```

# 2- EDA

Showing the first 5 rows of the dataset:


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_index</th>
      <th>accident_year</th>
      <th>accident_reference</th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>police_force</th>
      <th>accident_severity</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>date</th>
      <th>day_of_week</th>
      <th>time</th>
      <th>local_authority_district</th>
      <th>local_authority_ons_district</th>
      <th>local_authority_highway</th>
      <th>first_road_class</th>
      <th>first_road_number</th>
      <th>road_type</th>
      <th>speed_limit</th>
      <th>junction_detail</th>
      <th>junction_control</th>
      <th>second_road_class</th>
      <th>second_road_number</th>
      <th>pedestrian_crossing_human_control</th>
      <th>pedestrian_crossing_physical_facilities</th>
      <th>light_conditions</th>
      <th>weather_conditions</th>
      <th>road_surface_conditions</th>
      <th>special_conditions_at_site</th>
      <th>carriageway_hazards</th>
      <th>urban_or_rural_area</th>
      <th>did_police_officer_attend_scene_of_accident</th>
      <th>trunk_road_flag</th>
      <th>lsoa_of_accident_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016010000005</td>
      <td>2016</td>
      <td>010000005</td>
      <td>519310.0</td>
      <td>188730.0</td>
      <td>-0.279323</td>
      <td>51.584754</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>2</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>02:30</td>
      <td>Brent</td>
      <td>Brent</td>
      <td>Brent</td>
      <td>A</td>
      <td>4006</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights unlit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01000543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016010000006</td>
      <td>2016</td>
      <td>010000006</td>
      <td>551920.0</td>
      <td>174560.0</td>
      <td>0.184928</td>
      <td>51.449595</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>00:37</td>
      <td>Bexley</td>
      <td>Bexley</td>
      <td>Bexley</td>
      <td>A</td>
      <td>207</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Other junction</td>
      <td>Give way or uncontrolled</td>
      <td>Unclassified</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01000375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016010000008</td>
      <td>2016</td>
      <td>010000008</td>
      <td>505930.0</td>
      <td>183850.0</td>
      <td>-0.473837</td>
      <td>51.543563</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>01:25</td>
      <td>Hillingdon</td>
      <td>Hillingdon</td>
      <td>Hillingdon</td>
      <td>A</td>
      <td>4020</td>
      <td>Roundabout</td>
      <td>30.0</td>
      <td>Roundabout</td>
      <td>Give way or uncontrolled</td>
      <td>A</td>
      <td>4020.0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01033725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016010000016</td>
      <td>2016</td>
      <td>010000016</td>
      <td>527770.0</td>
      <td>168930.0</td>
      <td>-0.164442</td>
      <td>51.404958</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>09:15</td>
      <td>Merton</td>
      <td>Merton</td>
      <td>Merton</td>
      <td>A</td>
      <td>217</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>T or staggered junction</td>
      <td>Auto traffic signal</td>
      <td>A</td>
      <td>217.0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01003379</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016010000018</td>
      <td>2016</td>
      <td>010000018</td>
      <td>510740.0</td>
      <td>177230.0</td>
      <td>-0.406580</td>
      <td>51.483139</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>2</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>07:53</td>
      <td>Hounslow</td>
      <td>Hounslow</td>
      <td>Hounslow</td>
      <td>A</td>
      <td>312</td>
      <td>Dual carriageway</td>
      <td>40.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01002583</td>
    </tr>
  </tbody>
</table>
</div>



Getting the size of the dataset:


```python
df.shape
```




    (136621, 36)



Showing statistics that summarize the central tendency of the variables:


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_year</th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>speed_limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>136621.0</td>
      <td>136614.000000</td>
      <td>1.366140e+05</td>
      <td>136614.000000</td>
      <td>136614.000000</td>
      <td>136621.000000</td>
      <td>136621.000000</td>
      <td>136584.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2016.0</td>
      <td>448699.363169</td>
      <td>2.883354e+05</td>
      <td>-1.304881</td>
      <td>52.482399</td>
      <td>1.848179</td>
      <td>1.327644</td>
      <td>37.943683</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>95230.253169</td>
      <td>1.570588e+05</td>
      <td>1.398947</td>
      <td>1.414390</td>
      <td>0.710117</td>
      <td>0.789296</td>
      <td>14.041669</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2016.0</td>
      <td>76702.000000</td>
      <td>1.107500e+04</td>
      <td>-7.389809</td>
      <td>49.919716</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2016.0</td>
      <td>386355.750000</td>
      <td>1.764248e+05</td>
      <td>-2.204357</td>
      <td>51.473779</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2016.0</td>
      <td>454126.000000</td>
      <td>2.374855e+05</td>
      <td>-1.201205</td>
      <td>52.025165</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.0</td>
      <td>527660.000000</td>
      <td>3.897438e+05</td>
      <td>-0.159708</td>
      <td>53.401675</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.0</td>
      <td>655256.000000</td>
      <td>1.178623e+06</td>
      <td>1.757858</td>
      <td>60.490191</td>
      <td>16.000000</td>
      <td>58.000000</td>
      <td>70.000000</td>
    </tr>
  </tbody>
</table>
</div>



Getting a short summary of our dataset:


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 136621 entries, 0 to 136620
    Data columns (total 36 columns):
     #   Column                                       Non-Null Count   Dtype  
    ---  ------                                       --------------   -----  
     0   accident_index                               136621 non-null  object 
     1   accident_year                                136621 non-null  int64  
     2   accident_reference                           136621 non-null  object 
     3   location_easting_osgr                        136614 non-null  float64
     4   location_northing_osgr                       136614 non-null  float64
     5   longitude                                    136614 non-null  float64
     6   latitude                                     136614 non-null  float64
     7   police_force                                 136621 non-null  object 
     8   accident_severity                            136621 non-null  object 
     9   number_of_vehicles                           136621 non-null  int64  
     10  number_of_casualties                         136621 non-null  int64  
     11  date                                         136621 non-null  object 
     12  day_of_week                                  136621 non-null  object 
     13  time                                         136621 non-null  object 
     14  local_authority_district                     136621 non-null  object 
     15  local_authority_ons_district                 136621 non-null  object 
     16  local_authority_highway                      136621 non-null  object 
     17  first_road_class                             136621 non-null  object 
     18  first_road_number                            136621 non-null  object 
     19  road_type                                    135222 non-null  object 
     20  speed_limit                                  136584 non-null  float64
     21  junction_detail                              136621 non-null  object 
     22  junction_control                             136621 non-null  object 
     23  second_road_class                            136621 non-null  object 
     24  second_road_number                           79614 non-null   object 
     25  pedestrian_crossing_human_control            136621 non-null  object 
     26  pedestrian_crossing_physical_facilities      136621 non-null  object 
     27  light_conditions                             136621 non-null  object 
     28  weather_conditions                           132808 non-null  object 
     29  road_surface_conditions                      136621 non-null  object 
     30  special_conditions_at_site                   136621 non-null  object 
     31  carriageway_hazards                          136621 non-null  object 
     32  urban_or_rural_area                          136621 non-null  object 
     33  did_police_officer_attend_scene_of_accident  136621 non-null  object 
     34  trunk_road_flag                              136621 non-null  object 
     35  lsoa_of_accident_location                    136621 non-null  object 
    dtypes: float64(5), int64(3), object(28)
    memory usage: 37.5+ MB
    

Checking if there are any duplicated rows in the dataset:


```python
df.duplicated().sum()
```




    0



Study of relationships between variables:


```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x24835124b80>




    
![png](Milestone%201%20template_files/Milestone%201%20template_21_1.png)
    


Knowing which variables are strongly correlated with each other:


```python
corrmat = df.corr()
plt.figure(figsize = (16,16))
hm = sns.heatmap(corrmat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 linewidths=0.5,
                 annot_kws={'size': 15},
                 cmap="Spectral_r")
plt.show()
```

    C:\Users\lenovo\AppData\Local\Temp\ipykernel_3936\3101994207.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      corrmat = df.corr()
    


    
![png](Milestone%201%20template_files/Milestone%201%20template_23_1.png)
    


## Observations:

1- The dataset contains the data of accidents.
2- The number of accidents in the dataset is 136621 each of them has an index and another 35 piece of information that describe the accident.
3- All the accidents in the dataset happened in 2016.
4- At least one vehicle got affected in each accident and at most 16 vehicle got affected.
5- At least one casualty in each accident and at most 58 ones.
6- speed limit of vehicles in the accidents goes from 20 to 70.
7- Some of columns cotain null values which are (location_easting_osgr, location_northing_osgr, longitude, latitude, speed_limit, second_road_number, weather_conditions).
8- The information about second_road_number is missing alot in the dataset.
9- There are no duplicated rows in the dataset.
10- location_easting_osgr and longitude are very correlated to each other.
11- location_northing_osgr and latitude are very correlated to each other.
12- number_of_vehicles and number_of_casualties are strongly positive correlated.
13- speed_limit and number_of_casualties are strongly positive correlated.

What the effect of the weekends on the number of accidents?
The number of accidents is the least during the weekends relative to the number of accidents in the other days.


```python
def CountPlot(columnName):
  sns.countplot(data=df, x=columnName)
  plt.title('Bar Plot for '+ columnName)
  plt.show()

def BoxPlot(columnName):
  sns.boxplot(x=columnName, data=df)
  plt.title('Boxplot of '+columnName)
  plt.show()

def CountPlot2(columnName1, columnName2):
  sns.countplot(data=df, x=columnName1, hue=columnName2)
  plt.title('Bar Plot for '+columnName1)
  plt.show()

def ScatterSum(columnName1, columnName2):
  count = df.groupby([columnName1])[columnName2].sum()
  plt.xlabel(columnName1)
  plt.ylabel(columnName2)
  plt.title(columnName2+' VS. '+columnName1)
  plt.scatter(count.index,count)
  plt.show()

def ScatterAverage(columnName1, columnName2):
  average = df.groupby([columnName1])[columnName2].sum()/ df.groupby([columnName1])[columnName2].count()
  plt.xlabel(columnName1)
  plt.ylabel(columnName2)
  plt.title(columnName2+' VS. '+columnName1)
  plt.scatter(average.index,average)
  plt.show()

```


```python
CountPlot("day_of_week")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_27_0.png)
    


How does the speed limit affect the accidents?
The high speed limits don't cause high number of accidents as the most accidents are ranging between 20 to 50.


```python
BoxPlot("speed_limit")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_29_0.png)
    


What the number of vehicles in the accidents?
The number of vehicles is ranging between 1 - 3 however some accidents have up to 16 vehicles.


```python
BoxPlot("number_of_vehicles")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_31_0.png)
    


What is the effect of the road and area type on the number of accidents?
The most of the accidents happened in Urban areas and in single carriage ways.


```python
CountPlot2("road_type", "urban_or_rural_area")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_33_0.png)
    


What is the effect of the light conditions and weather conditions on the number of accidents?
The most accidents happened during the day and with fine weather.


```python
CountPlot2("light_conditions", "weather_conditions")

```


    
![png](Milestone%201%20template_files/Milestone%201%20template_35_0.png)
    


What is the relation between the number of casualties and the number of vehicles in the accidents?
Almost all casualties are in 1 or 2 vehicles accidents.


```python
ScatterSum("number_of_vehicles","number_of_casualties")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_37_0.png)
    


How does the speed limit affect the number of casualties?
Almost all casualties happened in 30 and 60 speed limits.


```python
ScatterSum("speed_limit","number_of_casualties")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_39_0.png)
    


What is the effect of junction control on the number casualties?
Almost all accidents happened in uncontrolled junctions.


```python
ScatterSum("junction_control","number_of_casualties")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_41_0.png)
    


How does the road type affect the average of number of casualties?
The highest average of number of casualties is in the accidents with missing data about road type.


```python
ScatterAverage("road_type", "number_of_casualties")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_43_0.png)
    


How does the junction affect the number of vehicles in the accidents?
The slip roads have the highest average of number of vehicles in the accidents.


```python
ScatterAverage("junction_detail", "number_of_vehicles")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_45_0.png)
    


# 3 - Cleaning Data

## Observing missing Data

Let's look at how many null values we have in the dataset


```python
def get_nan_count(dataset: pd.DataFrame, percentage=False, nan_word=None):
    if(nan_word == None):
        null_cnt = dataset.isnull().sum()
    else:
        null_cnt = (dataset == nan_word).sum()
    null_cnt = null_cnt[null_cnt > 0]
    if(percentage):
        null_cnt = null_cnt * 100 / len(dataset)
    return null_cnt

dataset = df
get_nan_count(dataset), get_nan_count(dataset, percentage=True)
```




    (location_easting_osgr         7
     location_northing_osgr        7
     longitude                     7
     latitude                      7
     road_type                  1399
     speed_limit                  37
     second_road_number        57007
     weather_conditions         3813
     dtype: int64,
     location_easting_osgr      0.005124
     location_northing_osgr     0.005124
     longitude                  0.005124
     latitude                   0.005124
     road_type                  1.024001
     speed_limit                0.027082
     second_road_number        41.726382
     weather_conditions         2.790933
     dtype: float64)



### Missing location values

For location columns [location_easting_osgr, location_northing_osgr, longitude, latitude] we have 7 missing records, let's look at them


```python
dataset[dataset['longitude'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_index</th>
      <th>accident_year</th>
      <th>accident_reference</th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>police_force</th>
      <th>accident_severity</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>date</th>
      <th>day_of_week</th>
      <th>time</th>
      <th>local_authority_district</th>
      <th>local_authority_ons_district</th>
      <th>local_authority_highway</th>
      <th>first_road_class</th>
      <th>first_road_number</th>
      <th>road_type</th>
      <th>speed_limit</th>
      <th>junction_detail</th>
      <th>junction_control</th>
      <th>second_road_class</th>
      <th>second_road_number</th>
      <th>pedestrian_crossing_human_control</th>
      <th>pedestrian_crossing_physical_facilities</th>
      <th>light_conditions</th>
      <th>weather_conditions</th>
      <th>road_surface_conditions</th>
      <th>special_conditions_at_site</th>
      <th>carriageway_hazards</th>
      <th>urban_or_rural_area</th>
      <th>did_police_officer_attend_scene_of_accident</th>
      <th>trunk_road_flag</th>
      <th>lsoa_of_accident_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62592</th>
      <td>2016210125234</td>
      <td>2016</td>
      <td>210125234</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Staffordshire</td>
      <td>Serious</td>
      <td>2</td>
      <td>1</td>
      <td>07/11/2016</td>
      <td>Monday</td>
      <td>06:00</td>
      <td>Stafford</td>
      <td>Stafford</td>
      <td>Staffordshire</td>
      <td>Unclassified</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Wet or damp</td>
      <td>None</td>
      <td>None</td>
      <td>Unallocated</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>62618</th>
      <td>2016210126357</td>
      <td>2016</td>
      <td>210126357</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Staffordshire</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>10/11/2016</td>
      <td>Thursday</td>
      <td>16:40</td>
      <td>Stoke-on-Trent</td>
      <td>Stoke-on-Trent</td>
      <td>Stoke-on-Trent</td>
      <td>A</td>
      <td>5272</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Raining no high winds</td>
      <td>Wet or damp</td>
      <td>None</td>
      <td>None</td>
      <td>Unallocated</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>83330</th>
      <td>2016400135181</td>
      <td>2016</td>
      <td>400135181</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bedfordshire</td>
      <td>Slight</td>
      <td>3</td>
      <td>3</td>
      <td>28/11/2016</td>
      <td>Monday</td>
      <td>07:55</td>
      <td>Bedford</td>
      <td>Bedford</td>
      <td>Bedford</td>
      <td>B</td>
      <td>645</td>
      <td>Single carriageway</td>
      <td>60.0</td>
      <td>T or staggered junction</td>
      <td>Give way or uncontrolled</td>
      <td>Unclassified</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Unallocated</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>129439</th>
      <td>2016930000146</td>
      <td>2016</td>
      <td>930000146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tayside</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>24/04/2016</td>
      <td>Sunday</td>
      <td>15:43</td>
      <td>Perth and Kinross</td>
      <td>Perth and Kinross</td>
      <td>Perth and Kinross</td>
      <td>A</td>
      <td>9</td>
      <td>Single carriageway</td>
      <td>60.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Unallocated</td>
      <td>Yes</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>132298</th>
      <td>2016961600599</td>
      <td>2016</td>
      <td>961600599</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Central</td>
      <td>Slight</td>
      <td>2</td>
      <td>3</td>
      <td>15/04/2016</td>
      <td>Friday</td>
      <td>13:30</td>
      <td>Stirling</td>
      <td>Stirling</td>
      <td>Stirling</td>
      <td>A</td>
      <td>872</td>
      <td>Data missing or out of range</td>
      <td>60.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>Data missing or out of range</td>
      <td>Data missing or out of range</td>
      <td>Data missing or out of range</td>
      <td>Data missing or out of range</td>
      <td>Data missing or out of range</td>
      <td>None</td>
      <td>Data missing or out of range</td>
      <td>Unallocated</td>
      <td>No</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>132431</th>
      <td>2016961601188</td>
      <td>2016</td>
      <td>961601188</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Central</td>
      <td>Serious</td>
      <td>2</td>
      <td>1</td>
      <td>03/08/2016</td>
      <td>Wednesday</td>
      <td>08:54</td>
      <td>Stirling</td>
      <td>Stirling</td>
      <td>Stirling</td>
      <td>A</td>
      <td>81</td>
      <td>Single carriageway</td>
      <td>60.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Raining no high winds</td>
      <td>Wet or damp</td>
      <td>None</td>
      <td>None</td>
      <td>Unallocated</td>
      <td>Yes</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>132527</th>
      <td>2016961601508</td>
      <td>2016</td>
      <td>961601508</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Central</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>26/09/2016</td>
      <td>Monday</td>
      <td>08:40</td>
      <td>Stirling</td>
      <td>Stirling</td>
      <td>Stirling</td>
      <td>A</td>
      <td>820</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
      <td>NaN</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Unallocated</td>
      <td>No</td>
      <td>Data missing or out of range</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>



Looking at these data, there's no common pattern or explanation between theses records. The missing data for location is MCAR.

### Missing road type values

Let's look at the road type different values and their count


```python
def visualize_cols_values_cnt(dataset, col_name, percentage=False, title=None):
    (dataset[col_name].value_counts(dropna=False) * ((100/len(dataset)) if percentage else 1) ).plot(kind='bar', xlabel=col_name, ylabel='count', title=title)
    

print('Values count of road type')
print(dataset['road_type'].value_counts(dropna=False))
visualize_cols_values_cnt(dataset, 'road_type')

```

    Values count of road type
    Single carriageway              101687
    Dual carriageway                 20117
    Roundabout                        8865
    One way street                    3117
    Slip road                         1435
    NaN                               1399
    Data missing or out of range         1
    Name: road_type, dtype: int64
    


    
![png](Milestone%201%20template_files/Milestone%201%20template_56_1.png)
    


Let's look at the road type missing values and there relation to the road class


```python
road_null_ds = dataset[dataset['road_type'].isna()]
visualize_cols_values_cnt(dataset, 'first_road_class', percentage=True, title="Whole dataset")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_58_0.png)
    



```python
visualize_cols_values_cnt(road_null_ds, 'first_road_class', percentage=True, title="Null road type")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_59_0.png)
    


We can see that the percentage of unclassified roads in the missing data is higher than in the total data. This suggests that there might be a correlation between the road class being unclassified and its type to be missing. 
It's only 1% of the data and we should not worry much about it.

### Missing speed values


```python
visualize_cols_values_cnt(dataset, 'speed_limit')
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_62_0.png)
    


There might be a relation between the speed limit being missing and the road class.


```python
visualize_cols_values_cnt(dataset[dataset.speed_limit.isna()], 'first_road_class', title="Roads where speed limit is null")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_64_0.png)
    


There are only 37 missing values for speed limit and most of these roads are unclassified so it might the reason that the speed limit is missing.

### Missing second road values

42% of the records have this feature missing which is a very big number. It can be explained by the fact that most of the accidents happen at only one road and not a crosswalk.

Let's look at the junction type values distribution to understand why all these values are missing.


```python
visualize_cols_values_cnt(dataset, 'junction_detail', percentage=True, title="Whole dataset")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_69_0.png)
    


The "Not a junction" value occurs around 42% of the dataset which is the same percentage as missing second road. This might be the reason for it to be missing.


```python

ds_null_2nd_road = dataset[dataset.second_road_number.isna()]
visualize_cols_values_cnt( ds_null_2nd_road, 'junction_detail', percentage=True,  title="Data where second road number is null")

```


    
![png](Milestone%201%20template_files/Milestone%201%20template_71_0.png)
    


From the above 2 charts we can see that almost 100% of the null values are coming from accidents happened not at a junction. This means that there's no second road at the first place. This is Missing At Random (MAR) type.   

### Missing weather condition values


```python
visualize_cols_values_cnt(dataset, 'weather_conditions', percentage=True)
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_74_0.png)
    


Because this is weather data. It might be justified that the weather was not reported during some accidents. It can be treated under the category of "Data missing or out of range".

### Observing that "Data missing or out of range" is userd instead of NaN

Some of the values are set to the value 'Data missing or out of range' instead of NaN


```python
nan_word = 'Data missing or out of range'
get_nan_count(dataset, percentage=True, nan_word=nan_word)
```




    road_type                                   0.000732
    junction_detail                             0.000732
    junction_control                           41.443848
    pedestrian_crossing_human_control           0.131019
    pedestrian_crossing_physical_facilities     0.122236
    light_conditions                            0.009515
    weather_conditions                          0.009515
    road_surface_conditions                     0.558479
    carriageway_hazards                         0.075391
    trunk_road_flag                             9.711538
    dtype: float64



Let's look at the junction control because it has 41% missing values. It looks like it has the same issue as the second road missing values when there's no junction.


```python
ds_null_junction_cont = dataset[dataset.junction_control == nan_word]
visualize_cols_values_cnt( ds_null_junction_cont, 'junction_detail', percentage=True,  title="Null junction control")
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_80_0.png)
    


This confirms our hypothesis since most of the missing values happens when there are no junctions which means there is no junction control. This is MAR.

Let's look at trunk_road_flag which has %10 missing


```python
visualize_cols_values_cnt(dataset, 'trunk_road_flag', percentage=True)
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_83_0.png)
    


I think it might be really missing for no reason (MCAR).

### None instead of NaN

Some of the values are set to the value 'None' instead of NaN


```python
nan_word = 'None'
get_nan_count(dataset, percentage=True, nan_word=nan_word)
```




    special_conditions_at_site    97.546497
    carriageway_hazards           98.021534
    dtype: float64



Let's look at the the values counts in these columns


```python
visualize_cols_values_cnt( dataset, 'special_conditions_at_site', percentage=True)
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_89_0.png)
    


The values here mean that none is means that there were no special conditions at the site.


```python
ds_null_junction_cont = dataset[dataset.junction_control == nan_word]
visualize_cols_values_cnt( dataset, 'carriageway_hazards', percentage=True)
```


    
![png](Milestone%201%20template_files/Milestone%201%20template_91_0.png)
    


The values here mean that none is means that there were no hazards on the road.

This concludes that None is not a missing value keyword

## Handling Missing data

Treat the other missing keywords as NaN


```python
ds_clean = dataset.replace(['Data missing or out of range', 'unknown (self reported)'], np.nan)
get_nan_count(ds_clean, percentage=True)
```




    location_easting_osgr                       0.005124
    location_northing_osgr                      0.005124
    longitude                                   0.005124
    latitude                                    0.005124
    road_type                                   1.024733
    speed_limit                                 0.027082
    junction_detail                             0.077587
    junction_control                           41.445312
    second_road_number                         41.726382
    pedestrian_crossing_human_control           0.135411
    pedestrian_crossing_physical_facilities     0.461862
    light_conditions                            0.009515
    weather_conditions                          2.800448
    road_surface_conditions                     0.562871
    special_conditions_at_site                  0.369636
    carriageway_hazards                         0.399646
    trunk_road_flag                             9.711538
    dtype: float64



Let's remove all the rows where some values are set to NaN in the columns with less than 1% missing values


```python
def remove_rows_with_nan_less_than(dataset: pd.DataFrame, threshold:float = 0):
    cnt = dataset.isna().sum()
    cnt = cnt[cnt > 0] 
    cnt = cnt * 100 / len(dataset)
    cnt = cnt[cnt < threshold]
    cols = cnt.index.to_list()
    return dataset.dropna(axis='index', subset=cols)
ds_clean = remove_rows_with_nan_less_than(ds_clean, threshold=1)
print(get_nan_count(ds_clean, percentage=True))
print('Removed percentage', (1 - len(ds_clean)/ len(dataset))*100)

```

    road_type              0.834417
    junction_control      41.275942
    second_road_number    41.751907
    weather_conditions     2.337699
    trunk_road_flag        9.798676
    dtype: float64
    Removed percentage 0.9639806471918644
    

We will handle the MCAR values that has higher percentage than 1% by replacing them with the mode. Because they are categorical and replacing by the mode won't affect much since their % are not much. These are road_type, weather_conditions, trunk_road_flag.


```python
def replace_nan_with_mode(dataset:pd.DataFrame, col_name):
    return dataset[col_name].fillna(dataset[col_name].mode()[0])


for col_name in ['road_type', 'weather_conditions', 'trunk_road_flag']:
    ds_clean[col_name] = replace_nan_with_mode(ds_clean, col_name)

get_nan_count(ds_clean, percentage=True)

```




    junction_control      41.275942
    second_road_number    41.751907
    dtype: float64



Since the junction control and the second road number are missing from nearly half of the records and their abscense is explained by the value of the junction detail being set to no junction (MAR), I will keep them to avoid the data being biased and I will replace their nan values with a universal "None" that indicates there's no value here.


```python
for col_name in ['junction_control', 'second_road_number']:
    ds_clean[col_name] = ds_clean[col_name].fillna('None')
get_nan_count(ds_clean, percentage=True)

```




    Series([], dtype: float64)



Since the values for the second road class contain -1 to indicate that there's no second road and thus there is no class, we will convert the -1 to None to be consistent with others.


```python
ds_clean.second_road_class = ds_clean.second_road_class.replace('-1', 'None')
```

### Findings and conclusions

There were not much missing values except for the junction situation. This was handled perfectly to reflect the missing values meaning. Now the dataset has no missing values.

## Handling unclean data

Let's inspect the values of some of the columns and ensure they are in good format.

The indicator features should have typically 2 values (true or false).


```python
ds_clean.nunique()
```




    accident_index                                 135304
    accident_year                                       1
    accident_reference                             135304
    location_easting_osgr                           94221
    location_northing_osgr                          96729
    longitude                                      129428
    latitude                                       128099
    police_force                                       51
    accident_severity                                   3
    number_of_vehicles                                 14
    number_of_casualties                               22
    date                                              366
    day_of_week                                         7
    time                                             1440
    local_authority_district                          380
    local_authority_ons_district                      381
    local_authority_highway                           208
    first_road_class                                    6
    first_road_number                                3322
    road_type                                           5
    speed_limit                                         6
    junction_detail                                     9
    junction_control                                    6
    second_road_class                                   7
    second_road_number                               2605
    pedestrian_crossing_human_control                   3
    pedestrian_crossing_physical_facilities             6
    light_conditions                                    5
    weather_conditions                                  8
    road_surface_conditions                             5
    special_conditions_at_site                          8
    carriageway_hazards                                 6
    urban_or_rural_area                                 2
    did_police_officer_attend_scene_of_accident         3
    trunk_road_flag                                     2
    lsoa_of_accident_location                       28602
    dtype: int64




```python
col_name = 'did_police_officer_attend_scene_of_accident'
ds_clean[col_name].value_counts()
```




    Yes                                                                         103633
    No                                                                           30574
    No - accident was reported using a self completion  form (self rep only)      1097
    Name: did_police_officer_attend_scene_of_accident, dtype: int64



It's more convenient to aggregate the two no options together.


```python
ds_clean[col_name] = ds_clean[col_name].replace('No - accident was reported using a self completion  form (self rep only)', 'No')
ds_clean[col_name].value_counts()
```




    Yes    103633
    No      31671
    Name: did_police_officer_attend_scene_of_accident, dtype: int64




```python
# for col_name in ['day_of_week', 'first_road_class', 'second_road_class', 'road_type', 'speed_limit', 'junction_detail', 'junction_control', 
# 'pedestrian_crossing_human_control', 'pedestrian_crossing_physical_facilities', 'light_conditions', 'weather_conditions', 'road_surface_conditions',
# 'special_conditions_at_site', 'carriageway_hazards', 'urban_or_rural_area', 'trunk_road_flag']:
#     print(ds_clean[col_name].value_counts())
#     print('-----------------------------------')
```

I looked at all the distinct values and they are all clean now.
I commented it to decrease the long scroll the notebook.

## Handling duplicate and useless data


```python
ds_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_index</th>
      <th>accident_year</th>
      <th>accident_reference</th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>police_force</th>
      <th>accident_severity</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>date</th>
      <th>day_of_week</th>
      <th>time</th>
      <th>local_authority_district</th>
      <th>local_authority_ons_district</th>
      <th>local_authority_highway</th>
      <th>first_road_class</th>
      <th>first_road_number</th>
      <th>road_type</th>
      <th>speed_limit</th>
      <th>junction_detail</th>
      <th>junction_control</th>
      <th>second_road_class</th>
      <th>second_road_number</th>
      <th>pedestrian_crossing_human_control</th>
      <th>pedestrian_crossing_physical_facilities</th>
      <th>light_conditions</th>
      <th>weather_conditions</th>
      <th>road_surface_conditions</th>
      <th>special_conditions_at_site</th>
      <th>carriageway_hazards</th>
      <th>urban_or_rural_area</th>
      <th>did_police_officer_attend_scene_of_accident</th>
      <th>trunk_road_flag</th>
      <th>lsoa_of_accident_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016010000005</td>
      <td>2016</td>
      <td>010000005</td>
      <td>519310.0</td>
      <td>188730.0</td>
      <td>-0.279323</td>
      <td>51.584754</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>2</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>02:30</td>
      <td>Brent</td>
      <td>Brent</td>
      <td>Brent</td>
      <td>A</td>
      <td>4006</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights unlit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01000543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016010000006</td>
      <td>2016</td>
      <td>010000006</td>
      <td>551920.0</td>
      <td>174560.0</td>
      <td>0.184928</td>
      <td>51.449595</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>00:37</td>
      <td>Bexley</td>
      <td>Bexley</td>
      <td>Bexley</td>
      <td>A</td>
      <td>207</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Other junction</td>
      <td>Give way or uncontrolled</td>
      <td>Unclassified</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01000375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016010000008</td>
      <td>2016</td>
      <td>010000008</td>
      <td>505930.0</td>
      <td>183850.0</td>
      <td>-0.473837</td>
      <td>51.543563</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>01:25</td>
      <td>Hillingdon</td>
      <td>Hillingdon</td>
      <td>Hillingdon</td>
      <td>A</td>
      <td>4020</td>
      <td>Roundabout</td>
      <td>30.0</td>
      <td>Roundabout</td>
      <td>Give way or uncontrolled</td>
      <td>A</td>
      <td>4020.0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01033725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016010000016</td>
      <td>2016</td>
      <td>010000016</td>
      <td>527770.0</td>
      <td>168930.0</td>
      <td>-0.164442</td>
      <td>51.404958</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>09:15</td>
      <td>Merton</td>
      <td>Merton</td>
      <td>Merton</td>
      <td>A</td>
      <td>217</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>T or staggered junction</td>
      <td>Auto traffic signal</td>
      <td>A</td>
      <td>217.0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01003379</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016010000018</td>
      <td>2016</td>
      <td>010000018</td>
      <td>510740.0</td>
      <td>177230.0</td>
      <td>-0.406580</td>
      <td>51.483139</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>2</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>07:53</td>
      <td>Hounslow</td>
      <td>Hounslow</td>
      <td>Hounslow</td>
      <td>A</td>
      <td>312</td>
      <td>Dual carriageway</td>
      <td>40.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01002583</td>
    </tr>
  </tbody>
</table>
</div>



The accident index and reference are both unique and identify an accident. We can remove the accident index because it's useless.

Let's also remove the accident year since all the values are 2016 and storing all of them would be a waste of memory.


```python
ds_clean = ds_clean.drop(['accident_index', 'accident_year'], axis=1)
ds_clean.columns

```




    Index(['accident_reference', 'location_easting_osgr', 'location_northing_osgr',
           'longitude', 'latitude', 'police_force', 'accident_severity',
           'number_of_vehicles', 'number_of_casualties', 'date', 'day_of_week',
           'time', 'local_authority_district', 'local_authority_ons_district',
           'local_authority_highway', 'first_road_class', 'first_road_number',
           'road_type', 'speed_limit', 'junction_detail', 'junction_control',
           'second_road_class', 'second_road_number',
           'pedestrian_crossing_human_control',
           'pedestrian_crossing_physical_facilities', 'light_conditions',
           'weather_conditions', 'road_surface_conditions',
           'special_conditions_at_site', 'carriageway_hazards',
           'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident',
           'trunk_road_flag', 'lsoa_of_accident_location'],
          dtype='object')



Let's see if there are duplicate values


```python
print('all columns', ds_clean.duplicated().sum())
print('accident reference', ds_clean.duplicated(['accident_reference']).sum())
all_cols_except_ref = ds_clean.columns.to_list()
all_cols_except_ref.remove('accident_reference')
print('all except reference', ds_clean.duplicated(all_cols_except_ref).sum())
```

    all columns 0
    accident reference 0
    all except reference 6
    

If we compared all the feature values for duplicates we will not find any. If we consider all the values except the reference, there exist 6 duplicate values which shall be removed.


```python
ds_clean = ds_clean.drop_duplicates(all_cols_except_ref)
print('all except reference', ds_clean.duplicated(all_cols_except_ref).sum())

```

    all except reference 0
    

## Observing outliers

There are only a couple of numerical features that might contain outliers. Let's have a look at some of their stats.


```python
ds_clean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>speed_limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>135298.000000</td>
      <td>1.352980e+05</td>
      <td>135298.000000</td>
      <td>135298.000000</td>
      <td>135298.000000</td>
      <td>135298.000000</td>
      <td>135298.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>448278.298216</td>
      <td>2.887785e+05</td>
      <td>-1.311062</td>
      <td>52.486391</td>
      <td>1.849111</td>
      <td>1.329332</td>
      <td>37.984967</td>
    </tr>
    <tr>
      <th>std</th>
      <td>95383.289072</td>
      <td>1.575378e+05</td>
      <td>1.401275</td>
      <td>1.418685</td>
      <td>0.710392</td>
      <td>0.791406</td>
      <td>14.061562</td>
    </tr>
    <tr>
      <th>min</th>
      <td>76702.000000</td>
      <td>1.107500e+04</td>
      <td>-7.389809</td>
      <td>49.919716</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>385850.250000</td>
      <td>1.763200e+05</td>
      <td>-2.212812</td>
      <td>51.473065</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>453622.500000</td>
      <td>2.380975e+05</td>
      <td>-1.209135</td>
      <td>52.029078</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>527457.000000</td>
      <td>3.903335e+05</td>
      <td>-0.162852</td>
      <td>53.407116</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>655256.000000</td>
      <td>1.178623e+06</td>
      <td>1.757858</td>
      <td>60.490191</td>
      <td>16.000000</td>
      <td>58.000000</td>
      <td>70.000000</td>
    </tr>
  </tbody>
</table>
</div>



The number of vehicles ranges from [1:16] which is reasonable range for cars involved in an accident. A range of [1:58] is also okay for the number of casualties. 

The speed limit is already categorized into a couple of values in range [20:70] which indicates no outliers.


```python
plt.boxplot(ds_clean.number_of_casualties)
ds_clean[ds_clean.number_of_casualties == 58]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_reference</th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>police_force</th>
      <th>accident_severity</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>date</th>
      <th>day_of_week</th>
      <th>time</th>
      <th>local_authority_district</th>
      <th>local_authority_ons_district</th>
      <th>local_authority_highway</th>
      <th>first_road_class</th>
      <th>first_road_number</th>
      <th>road_type</th>
      <th>speed_limit</th>
      <th>junction_detail</th>
      <th>junction_control</th>
      <th>second_road_class</th>
      <th>second_road_number</th>
      <th>pedestrian_crossing_human_control</th>
      <th>pedestrian_crossing_physical_facilities</th>
      <th>light_conditions</th>
      <th>weather_conditions</th>
      <th>road_surface_conditions</th>
      <th>special_conditions_at_site</th>
      <th>carriageway_hazards</th>
      <th>urban_or_rural_area</th>
      <th>did_police_officer_attend_scene_of_accident</th>
      <th>trunk_road_flag</th>
      <th>lsoa_of_accident_location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75626</th>
      <td>34NE09806</td>
      <td>504003.0</td>
      <td>290817.0</td>
      <td>-0.469132</td>
      <td>52.505308</td>
      <td>Northamptonshire</td>
      <td>Serious</td>
      <td>2</td>
      <td>58</td>
      <td>14/07/2016</td>
      <td>Thursday</td>
      <td>15:58</td>
      <td>East Northamptonshire</td>
      <td>East Northamptonshire</td>
      <td>Northamptonshire</td>
      <td>C</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>Single carriageway</td>
      <td>60.0</td>
      <td>Crossroads</td>
      <td>Give way or uncontrolled</td>
      <td>C</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Rural</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01027044</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](Milestone%201%20template_files/Milestone%201%20template_127_1.png)
    


Although there are many outliers according to the box plot. Looking at the accident info of the furthest point with 58 casualities makes sense as the severity is high.

To check the location outliars, I researched the uk min and max longitude [-8.62 : 1.77] and latitude [49.9 : 60.84] (src: https://gis.stackexchange.com/questions/152758/countries-latitude-and-longitude-range). Looking and the min and max of the values in the dataset implies that there are no outliers.

# 4 - Data transformation

## 4.1 - Discretization


### Converting date (Object) -> dateTime dataType



```python
dataset_copy=ds_clean.copy()
dataset_copy['date'] =  pd.to_datetime(dataset_copy['date'], format='%d/%m/%Y')
```


```python
dataset_copy.loc[:,"date"].head(2)

```




    0   2016-11-01
    1   2016-11-01
    Name: date, dtype: datetime64[ns]



### Discretizing the data into weeks according to the dates.



```python
ds_clean['Week number'] = dataset_copy['date'].dt.isocalendar().week

```


```python
ds_clean.loc[:,["date","Week number"]].tail(2)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>Week number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136619</th>
      <td>29/10/2016</td>
      <td>43</td>
    </tr>
    <tr>
      <th>136620</th>
      <td>25/12/2016</td>
      <td>51</td>
    </tr>
  </tbody>
</table>
</div>



## 4.11 - Findings and conclusions


This kind of discretisation is ***Equal frequency discretisation***


Equal width discretisation divides the scope of possible values into N bins of the same width.The width is determined by the range of values in the variable and the number of bins we wish to use to divide the variable:

width = (max value(Max number of days) - min value(min number of day) // 7

Having a discrete data not continues ones helps us to visualize and discover data more easily and efficiency


## 4.2 - Encoding
Encoding data is typically used to ensure the integrity and usability of data and is commonly used when data cannot be transferred in its current format between systems or applications.

We have three types of encoding techniques that we will use on our data, let's firstly mentioned them and why we need them.
That will help us in classifying the right encode system for each attribute
1. ***Label Encoding*** refers to converting the labels into a numeric form so as to convert them into the machine-readable form. Machine learning algorithms can then decide in a better way how those labels must be operated. It is an important pre-processing step for the structured dataset in supervised learning.
2. ***One-hot*** encoding ensures that machine learning does not assume that higher numbers are more important. For example, the value '8' is bigger than the value '1', but that does not make '8' more important than '1'. The same is true for words: the value 'laughter' is not more important than 'laugh'
3. ***One Hot Encoding of Frequent Categories*** is just One-hot encoding can be used to handle a large number of categories also. How does it do this? Suppose 200 categories are present in a feature then only those 10 categories which are the top 10 repeating categories will be chosen and one-hot encoding is applied to only those categories.

Let's start with ***Label Encoding***
and analysis which attributes we need and why
1. ***accident_severity*** : this feature is a categorical one and has only 3 values that describes how danger was the accident, so assigning numerical value to it will help the machine to identify its meaning
2. ***light_conditions*** : this feature is a categorical one and has only 5 values that describes the light intensity through the accident, so assigning numerical value to it will help the machine to identify its meaning
3. ***weather_conditions***: this feature is a categorical one and has only 8 values that describes the weather through the accident, so assigning numerical value to it will help the machine to identify its meaning
4. ***road_surface_conditions***: this feature is a categorical one and has only 5 values that describes the road surface through the accident, so assigning numerical value to it will help the machine to identify its meaning

***We will give them values based on their meaning to avoid the machine encode which can lead to unreal numeric value***


```python
print(ds_clean["accident_severity"].unique())

```

    ['Slight' 'Serious' 'Fatal']
    


```python
print(ds_clean["light_conditions"].unique())

```

    ['Darkness - lights unlit' 'Darkness - lights lit' 'Daylight'
     'Darkness - no lighting' 'Darkness - lighting unknown']
    


```python
print(ds_clean["weather_conditions"].unique())

```

    ['Fine no high winds' 'Raining no high winds' 'Fog or mist' 'Other'
     'Fine + high winds' 'Raining + high winds' 'Snowing no high winds'
     'Snowing + high winds']
    


```python
print(ds_clean["road_surface_conditions"].unique())

```

    ['Dry' 'Wet or damp' 'Flood over 3cm. deep' 'Frost or ice' 'Snow']
    


```python
print(ds_clean["did_police_officer_attend_scene_of_accident"].unique())

```

    ['Yes' 'No']
    


```python
print(ds_clean["trunk_road_flag"].unique())

```

    ['Non-trunk' 'Trunk (Roads managed by Highways England)']
    


```python
label_encoded_values={'accident_severity':{'Slight':1,'Serious':2,'Fatal':3},
                      'light_conditions':{'Daylight':1,'Darkness - lights lit':2, 'Darkness - lights unlit':3,'Darkness - no lighting':4,'Darkness - lighting unknown':0},
                      'weather_conditions':{'Fine no high winds':1, 'Fine + high winds':2, 'Raining no high winds':3,'Raining + high winds':4, 'Snowing no high winds':5,'Snowing + high winds':6,'Fog or mist':7 ,'Other':0},
                      'road_surface_conditions':{'Dry':1,'Wet or damp':2,'Flood over 3cm. deep':3,'Frost or ice':4,'Snow':5},
                      'did_police_officer_attend_scene_of_accident':{'Yes':1,"No":0},
                      'trunk_road_flag':{'Trunk (Roads managed by Highways England)':1,'Non-trunk':0}}
```


```python
# Function to encode categorical attributes in a given dataset as numbers
def number_encode_features(df,label_encoded_values_replacement):
    result = df.copy() # take a copy of the dataframe
    return result.replace(label_encoded_values_replacement)
```


```python
# Apply function defined above to accidents dataset
encoded_data = number_encode_features(ds_clean,label_encoded_values)

# Display last 5 records in transformed dataset to verify numerical transformation
encoded_data.loc[:,["accident_severity","light_conditions",'weather_conditions',"road_surface_conditions","did_police_officer_attend_scene_of_accident","trunk_road_flag"]].tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_severity</th>
      <th>light_conditions</th>
      <th>weather_conditions</th>
      <th>road_surface_conditions</th>
      <th>did_police_officer_attend_scene_of_accident</th>
      <th>trunk_road_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136616</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136617</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136618</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136619</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136620</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#this function working on saving the label encoded dataset columns in a new csv
def save_label_encoded_columns_to_csv(column1_name,column2_name,column1_df,column2_df,csv_name):
    temp= pd.DataFrame()
    temp[column1_name]=column1_df
    temp[column2_name]=column2_df
    temp.to_csv(csv_name)
```


```python
save_label_encoded_columns_to_csv("accident_severity","accident_severity_encoded",ds_clean.accident_severity,encoded_data.accident_severity,'./label_encoded/'+"accident_severity_label_encoded")
save_label_encoded_columns_to_csv("light_conditions","light_conditions_encoded",ds_clean.light_conditions,encoded_data.light_conditions,'./label_encoded/'+"light_conditions_label_encoded")
save_label_encoded_columns_to_csv("weather_conditions","weather_conditions_encoded",ds_clean.weather_conditions,encoded_data.weather_conditions,'./label_encoded/'+"weather_conditions_label_encoded")
save_label_encoded_columns_to_csv("road_surface_conditions","road_surface_conditions_encoded",ds_clean.road_surface_conditions,encoded_data.road_surface_conditions,'./label_encoded/'+"road_surface_conditions_label_encoded")
save_label_encoded_columns_to_csv("did_police_officer_attend_scene_of_accident","did_police_officer_attend_scene_of_accident_encoded",ds_clean.did_police_officer_attend_scene_of_accident,encoded_data.did_police_officer_attend_scene_of_accident,'./label_encoded/'+"did_police_officer_attend_scene_of_accident_label_encoded")
save_label_encoded_columns_to_csv("trunk_road_flag","trunk_road_flag_encoded",ds_clean.trunk_road_flag,encoded_data.trunk_road_flag,'./label_encoded/'+"trunk_road_flag_label_encoded")
```

Now let's jump to the other type of encoding which is ***one_hot_encoding*** we will apply it on other categorical data that we didn't include in the label encoding


***First Step***: we should first define which columns we will apply one hot encoding on it, we will do that by including dtype='object' only


```python
encoded_data=encoded_data.drop(['accident_reference', 'lsoa_of_accident_location','date','time'], axis=1)
```


```python
def get_categorical_data(df):
    return df.select_dtypes(include = "object").columns.tolist()
```


```python
print(get_categorical_data(encoded_data))

```

    ['police_force', 'day_of_week', 'local_authority_district', 'local_authority_ons_district', 'local_authority_highway', 'first_road_class', 'first_road_number', 'road_type', 'junction_detail', 'junction_control', 'second_road_class', 'second_road_number', 'pedestrian_crossing_human_control', 'pedestrian_crossing_physical_facilities', 'special_conditions_at_site', 'carriageway_hazards', 'urban_or_rural_area']
    

***Second Step***: we will loop over the selected features and apply one hot encode only to the top 10 freq values in them


```python
# calculate top categories of variable in a dataframe
def calculate_top_categories(df, variable, how_many):
    return [
        x for x in df[variable].value_counts().sort_values(
            ascending=False).head(how_many).index
    ]

#apply one hot encode to the top_x_labels of variable of a dataframe and save them in a csv file on disk
def one_hot_encode(df, variable, top_x_labels):
    df_temp = pd.DataFrame()
    for label in top_x_labels:
        df_temp[label] = np.where(
            df[variable] == label, 1, 0)
    df_temp.to_csv(r"./OneHotEncoded/"+variable, sep='\t')

def apply_one_hot_encode_on_df(df):
    features=get_categorical_data(df)
    for feature in features:
        top_x_labels=calculate_top_categories(df,feature,10) #10 maximum
        one_hot_encode(df,feature,top_x_labels)
```


```python
apply_one_hot_encode_on_df(encoded_data.copy())

```

## 4.22 - Findings and conlcusions


***Label encoding***
- Pros:
    1. Straightforward to implement
    2. Does not expand the feature space
- cons:
  should be used with high precautions, we should try to get the best features that suitable for it and not run a general encoding to avoid such this mistake:
The problem using the number is that they introduce relation/comparison between them. Let’s consider another column named ‘Safety Level’ that has the values none < low < medium < high < very high. Performing label encoding of this column also induces order/precedence in number, but in the right way. Here the numerical order does not look out-of-box and it makes sense if the algorithm interprets safety order 0 < 1 < 2 < 3 < 4 i.e. none < low < medium < high < very high.

***One hot encoding***
- Pros:
    1. Does not add any information that may make the variable more predictive
    2. Does not keep the information of the ignored labels
- Cons:
    1. Memory consumption for large attributes
    2. When we're applying a threshold to avoid the first one, we have another limitations like:
       - Does not add any information that may make the variable more predictive
        - Does not keep the information of the ignored labels

## 4.3 - Normalisation


```python
fig, ax=plt.subplots(1,2)
sns.distplot(ds_clean["longitude"], ax=ax[0])
ax[0].set_title("longitude scale data")
sns.distplot(ds_clean["latitude"], ax=ax[1])
ax[1].set_title("latitude scale data")
plt.show()
```

    C:\Users\lenovo\AppData\Local\Temp\ipykernel_3936\3807337554.py:2: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(ds_clean["longitude"], ax=ax[0])
    C:\Users\lenovo\AppData\Local\Temp\ipykernel_3936\3807337554.py:4: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(ds_clean["latitude"], ax=ax[1])
    


    
![png](Milestone%201%20template_files/Milestone%201%20template_167_1.png)
    



```python
ds_clean.loc[:,["longitude","latitude"]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>135298.000000</td>
      <td>135298.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.311062</td>
      <td>52.486391</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.401275</td>
      <td>1.418685</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.389809</td>
      <td>49.919716</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-2.212812</td>
      <td>51.473065</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.209135</td>
      <td>52.029078</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-0.162852</td>
      <td>53.407116</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.757858</td>
      <td>60.490191</td>
    </tr>
  </tbody>
</table>
</div>



As we can notice the longitude and latitude have different scale, so we should rescale them into the same range to have the same effect when feeding
into a machine learning model

Now we will renormalize them using Scaling technique to have a range between 0 and 1



```python
longitude_after_scaling,latitude_after_scaling =  MinMaxScaler().fit_transform(ds_clean[["longitude"]]),MinMaxScaler().fit_transform(ds_clean[["latitude"]])
```

## 4.31 - Findings and conclusions



```python
fig, ax=plt.subplots(1,2)
sns.distplot(longitude_after_scaling, ax=ax[0])
ax[0].set_title("longitude after scale")
sns.distplot(latitude_after_scaling, ax=ax[1])
ax[1].set_title("latitude after scale")
plt.show()
```

    C:\Users\lenovo\AppData\Local\Temp\ipykernel_3936\97550941.py:2: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(longitude_after_scaling, ax=ax[0])
    C:\Users\lenovo\AppData\Local\Temp\ipykernel_3936\97550941.py:4: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(latitude_after_scaling, ax=ax[1])
    


    
![png](Milestone%201%20template_files/Milestone%201%20template_173_1.png)
    


***Scaling in a very important step in norimlizing data***
- Scaling is used to make all features contribute the same amount in prediction.
- Scaling makes the algorithms converge faster since it transform the variable space in to a much smaller range.


```python
#saving new scaling values in a new data frame
temp_df=pd.DataFrame()
temp_df["longitude"]=longitude_after_scaling.tolist()
temp_df["latitude"]=latitude_after_scaling.tolist()
save_label_encoded_columns_to_csv("longitude","latitude",temp_df.longitude,temp_df.latitude,"./norimlization/longitude_latitude_after_scaling")
```

## 4.4 - Adding more columns


The data we have, we can make best use of it, by extracting new attributes from it.
***We could detect a new column that shows whether the accident happened on the weekend(1) or not(0)***


```python
ds_clean["accident_on_weekend"]= np.where((ds_clean["day_of_week"] =="Saturday") |  (ds_clean["day_of_week"] == "Sunday"), 1, 0)
ds_clean.loc[:,["day_of_week","accident_on_weekend"]].tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_of_week</th>
      <th>accident_on_weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136616</th>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136617</th>
      <td>Tuesday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136618</th>
      <td>Thursday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>136619</th>
      <td>Saturday</td>
      <td>1</td>
    </tr>
    <tr>
      <th>136620</th>
      <td>Sunday</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



***We can also extract when the accident happened using the feature time to detect weather Morning, Afternoon, Evening and Night***


```python
# converting time Object format to datetime format
ds_clean['time'] = pd.to_datetime(ds_clean['time'],format= '%H:%M')
ds_clean.loc[ds_clean['time'].dt.hour<12, ['day_time']] = 'Morning'
ds_clean.loc[(ds_clean['time'].dt.hour>=12) & (ds_clean['time'].dt.hour<19), ['day_time']] = 'Afternoon'
ds_clean.loc[(ds_clean['time'].dt.hour>=19) & (ds_clean['time'].dt.hour<19), ['day_time']] = 'Evening'
ds_clean.loc[(ds_clean['time'].dt.hour>=19) & (ds_clean['time'].dt.hour<24), ['day_time']] = 'Night'
ds_clean['time'] = pd.to_datetime(ds_clean['time'],format= '%H:%M:%S').dt.time
ds_clean.loc[:,["time","day_time"]].tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>day_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136616</th>
      <td>06:45:00</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>136617</th>
      <td>16:45:00</td>
      <td>Afternoon</td>
    </tr>
    <tr>
      <th>136618</th>
      <td>07:10:00</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>136619</th>
      <td>20:00:00</td>
      <td>Night</td>
    </tr>
    <tr>
      <th>136620</th>
      <td>12:30:00</td>
      <td>Afternoon</td>
    </tr>
  </tbody>
</table>
</div>



## 4.41 - Findings and concluisons


Feature Extraction aims to reduce the number of features in a dataset by creating new features from the existing ones (and then discarding the original features). These new reduced set of features should then be able to summarize most of the information contained in the original set of features.

## 4.5 - Csv file for lookup



```python
# let's save our final dataset into a new one after cleaning and applying transormation
ds_clean.to_csv("final_accidents_uk")
```

## 5- Exporting the dataframe to a csv file or parquet



```python
df = pd.read_csv('final_accidents_uk',index_col=0)
df.astype(str).to_parquet('final_accidents_uk.parquet')
```

    C:\Users\lenovo\AppData\Local\Temp\ipykernel_3936\2188962871.py:1: DtypeWarning: Columns (34) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv('final_accidents_uk',index_col=0)
    


```python
df=pd.read_parquet('final_accidents_uk.parquet')
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident_reference</th>
      <th>location_easting_osgr</th>
      <th>location_northing_osgr</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>police_force</th>
      <th>accident_severity</th>
      <th>number_of_vehicles</th>
      <th>number_of_casualties</th>
      <th>date</th>
      <th>day_of_week</th>
      <th>time</th>
      <th>local_authority_district</th>
      <th>local_authority_ons_district</th>
      <th>local_authority_highway</th>
      <th>first_road_class</th>
      <th>first_road_number</th>
      <th>road_type</th>
      <th>speed_limit</th>
      <th>junction_detail</th>
      <th>junction_control</th>
      <th>second_road_class</th>
      <th>second_road_number</th>
      <th>pedestrian_crossing_human_control</th>
      <th>pedestrian_crossing_physical_facilities</th>
      <th>light_conditions</th>
      <th>weather_conditions</th>
      <th>road_surface_conditions</th>
      <th>special_conditions_at_site</th>
      <th>carriageway_hazards</th>
      <th>urban_or_rural_area</th>
      <th>did_police_officer_attend_scene_of_accident</th>
      <th>trunk_road_flag</th>
      <th>lsoa_of_accident_location</th>
      <th>Week number</th>
      <th>accident_on_weekend</th>
      <th>day_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>010000005</td>
      <td>519310.0</td>
      <td>188730.0</td>
      <td>-0.279323</td>
      <td>51.584754</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>2</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>02:30:00</td>
      <td>Brent</td>
      <td>Brent</td>
      <td>Brent</td>
      <td>A</td>
      <td>4006</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights unlit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01000543</td>
      <td>44</td>
      <td>0</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>010000006</td>
      <td>551920.0</td>
      <td>174560.0</td>
      <td>0.184928</td>
      <td>51.449595</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>00:37:00</td>
      <td>Bexley</td>
      <td>Bexley</td>
      <td>Bexley</td>
      <td>A</td>
      <td>207</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>Other junction</td>
      <td>Give way or uncontrolled</td>
      <td>Unclassified</td>
      <td>first_road_class is C or Unclassified. These r...</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01000375</td>
      <td>44</td>
      <td>0</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>2</th>
      <td>010000008</td>
      <td>505930.0</td>
      <td>183850.0</td>
      <td>-0.473837</td>
      <td>51.543563</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>01:25:00</td>
      <td>Hillingdon</td>
      <td>Hillingdon</td>
      <td>Hillingdon</td>
      <td>A</td>
      <td>4020</td>
      <td>Roundabout</td>
      <td>30.0</td>
      <td>Roundabout</td>
      <td>Give way or uncontrolled</td>
      <td>A</td>
      <td>4020.0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Darkness - lights lit</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01033725</td>
      <td>44</td>
      <td>0</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>010000016</td>
      <td>527770.0</td>
      <td>168930.0</td>
      <td>-0.164442</td>
      <td>51.404958</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>1</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>09:15:00</td>
      <td>Merton</td>
      <td>Merton</td>
      <td>Merton</td>
      <td>A</td>
      <td>217</td>
      <td>Single carriageway</td>
      <td>30.0</td>
      <td>T or staggered junction</td>
      <td>Auto traffic signal</td>
      <td>A</td>
      <td>217.0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01003379</td>
      <td>44</td>
      <td>0</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>010000018</td>
      <td>510740.0</td>
      <td>177230.0</td>
      <td>-0.40658</td>
      <td>51.483139</td>
      <td>Metropolitan Police</td>
      <td>Slight</td>
      <td>2</td>
      <td>1</td>
      <td>01/11/2016</td>
      <td>Tuesday</td>
      <td>07:53:00</td>
      <td>Hounslow</td>
      <td>Hounslow</td>
      <td>Hounslow</td>
      <td>A</td>
      <td>312</td>
      <td>Dual carriageway</td>
      <td>40.0</td>
      <td>Not at junction or within 20 metres</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None within 50 metres</td>
      <td>No physical crossing facilities within 50 metres</td>
      <td>Daylight</td>
      <td>Fine no high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>Urban</td>
      <td>Yes</td>
      <td>Non-trunk</td>
      <td>E01002583</td>
      <td>44</td>
      <td>0</td>
      <td>Morning</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
