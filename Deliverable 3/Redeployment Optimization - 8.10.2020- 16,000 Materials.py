#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pulp import *
import math
import timeit
import qgrid
import warnings;


# In[2]:


#Timer used to calculate runtime of model
start = timeit.default_timer()
#Remove warning for pasting material
warnings.filterwarnings('ignore');


# In[3]:


shipping_data = pd.read_csv('C:/Users/wenselbp/Desktop/Redeployment optimization/Python Files/full Shipping Lane DF.csv', index_col=['Receiving_Plant','Shipping_Plant', 'Mode'])


# In[4]:


shipping_df = shipping_data.drop(['Lane Code','Lane'], axis = 1)


# In[5]:


#qgrid.show_grid(shipping_df)


# In[6]:


material_data = pd.read_csv('C:/Users/wenselbp/Desktop/Redeployment optimization/Python Files/16000 Material DF.csv', index_col=['Node'])


# In[36]:


material_data.describe()


# In[8]:


optimal_shipment_df = pd.DataFrame(columns=['Material','shipment', 'shipping_lane_cost', 'shortage_cost','total_cost'])


# In[9]:


for unique_material in material_data['Material'].unique():
        material_df = material_data[material_data['Material']==unique_material]

        #Variables
        shipment = pulp.LpVariable.dicts("shipment",
                                         ((Receiving_Plant, Shipping_Plant, Mode) for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index),
                                        lowBound=0,
                                        cat='Integer')

        shortage_cost = pulp.LpVariable.dicts("shortage_cost",
                                         ((Receiving_Plant, Shipping_Plant, Mode) for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index),
                                        lowBound=0,
                                        cat='Integer')
        #Shortage Cost Dict
        for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index:
            try:
                if (material_df.loc[(Receiving_Plant), 'Need_in_days'] - shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode),'Transit Time']) < 0:
                    shortage_cost[Receiving_Plant, Shipping_Plant, Mode] = abs((material_df.loc[(Receiving_Plant), 'Need_in_days'] - shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode),'Transit Time']) * material_df.loc[(Receiving_Plant), 'daily_shortage_cost'])
                else:
                    shortage_cost[Receiving_Plant, Shipping_Plant, Mode] = 0
            except:
                shortage_cost[Receiving_Plant, Shipping_Plant, Mode] = 9999999
            



        #LP Problem    
        model = pulp.LpProblem("cost_min_problem", LpMinimize)

        model += pulp.lpSum(
            [shipment[Receiving_Plant, Shipping_Plant, Mode] * shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate']
        for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index]   

            + [shipment[Receiving_Plant, Shipping_Plant, Mode] * shortage_cost[(Receiving_Plant, Shipping_Plant, Mode)]        
        for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index]
        )    

        #Constraints   
        #Shipment from any lane must be equal to demand
        for node in material_df.index:
            model += shipment[(node, 1020,'IM')] + shipment[(node, 1100,'IM')] + shipment[(node, 1415,'IM')] + shipment[(node, 1645,'IM')] + shipment[(node, 1020,'TL')] + shipment[(node, 1100,'TL')] + shipment[(node, 1415,'TL')] + shipment[(node, 1645,'TL')] == material_df.loc[node, 'Projected_Shortage_Qty']


        #Shipment from any lane must be less or eqaul to the available supply at that node
        for node in material_df.index:
            model += shipment[(1020, node,'IM')] + shipment[(1020, node,'TL')] <= material_df.loc[node, 'Available Supply']
            model += shipment[(1100, node,'IM')] + shipment[(1100, node,'TL')] <= material_df.loc[node, 'Available Supply']
            model += shipment[(1415, node,'IM')] + shipment[(1415, node,'TL')] <= material_df.loc[node, 'Available Supply']
            model += shipment[(1645, node,'IM')] + shipment[(1645, node,'TL')] <= material_df.loc[node, 'Available Supply']

        model.solve()


        model_output = []

        for Receiving_Plant, Shipping_Plant, Mode in shipment:

            columns = {
                'Receiving_Plant': Receiving_Plant,
                'Shipping_Plant': Shipping_Plant,
                'Mode': Mode,
                'shipment': shipment[(Receiving_Plant, Shipping_Plant, Mode)].varValue,
                'total_cost': shortage_cost[(Receiving_Plant, Shipping_Plant, Mode)] + shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate'],
                'shortage_cost': shortage_cost[(Receiving_Plant, Shipping_Plant, Mode)],
                'shipping_lane_cost': shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate'],


                #'cost': (shipment[(Receiving_Plant, Shipping_Plant, Mode)].varValue / 38000) * shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate'],
            }
            model_output.append(columns)
        output_df = pd.DataFrame.from_records(model_output).sort_values(['Receiving_Plant', 'Shipping_Plant', 'Mode'])
        output_df = output_df[output_df['shipping_lane_cost'] != 9999999] #removes phantom transportation combinations
        output_df = output_df[output_df['shortage_cost'] != 9999999] #removes phantom warehouse combinations
        output_df= output_df[['Receiving_Plant', 'Shipping_Plant', 'Mode', 'shipment', 'shipping_lane_cost', 'shortage_cost','total_cost']] #removed 'expected_days_late'


        optimal_shipment = output_df[output_df['shipment'] > 0]
        optimal_shipment['Material'] = unique_material
        optimal_shipment['model_status'] = LpStatus[model.status]
        optimal_shipment_df = optimal_shipment_df.append(optimal_shipment)
        
        


# In[10]:


optimal_shipment_df.set_index(['Receiving_Plant', 'Shipping_Plant', 'Mode','Material'], inplace=True)


# In[29]:


optimal_shipment_df.info()


# In[11]:


#Stop timer
stop = timeit.default_timer()
print('Run time: ', stop - start, 'seconds')


# In[33]:


print('Run time: ', (stop-start)/60, 'minutes')


# In[12]:


#Canonical Form for a single material
print(model)


# In[13]:


recommended_shipment_df = optimal_shipment_df[optimal_shipment_df['model_status']== "Optimal"]
infeasible_shipment_df = optimal_shipment_df[optimal_shipment_df['model_status']== "Infeasible"]


# In[14]:


#qgrid.show_grid(infeasible_shipment_df)


# In[15]:


#qgrid.show_grid(recommended_shipment_df)


# # Building Transports with the Optimal Shipment Quantity 

# In[16]:


truck_df = recommended_shipment_df.reset_index()


# In[17]:


material_data2 = pd.read_csv('C:/Users/wenselbp/Desktop/Redeployment optimization/Python Files/Material DF.csv')
material_data2['Node'] = material_data2['Node'].astype(float)


# In[18]:


material_data2 = material_data2.drop(['Material Text','Available Supply','Projected_Shortage_Qty','daily_shortage_cost'],axis=1)


# In[19]:


truck_df = pd.merge(truck_df, material_data2,  how= 'left', left_on=['Receiving_Plant','Material'], right_on = ['Node','Material'])


# In[20]:


truck_capacity = 38000


# In[21]:


qgrid.show_grid(truck_df)


# In[22]:


table = pd.pivot_table(truck_df,index=['Receiving_Plant', 'Shipping_Plant', 'Mode'],aggfunc={'shipment':np.sum,'shortage_cost':np.sum,'shipping_lane_cost':np.mean})


# In[23]:


table['total cost'] = (table['shipment']/truck_capacity).apply(np.ceil)*table['shipping_lane_cost'] + table['shortage_cost']


# In[24]:


qgrid.show_grid(table)


# In[25]:


print('Total cost of: ' + '${:,.2f}'.format(sum(table['total cost'])))


# In[ ]:




