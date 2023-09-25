import http.client
import json
import numpy as np

conn = http.client.HTTPSConnection("api-metoffice.apiconnect.ibmcloud.com")

headers = {
    'X-IBM-Client-Id': "013eb5f9d7a2ccb2f59adaf0f33bf0a4",
    'X-IBM-Client-Secret': "4f2f8c58deef9e8a224b62cc31ae8485",
    'accept': "application/json"
    }

conn.request("GET", "/v0/forecasts/point/hourly?excludeParameterMetadata=true&includeLocationName=true&latitude=53.234338&longitude=-2.305018", headers=headers)

res = conn.getresponse()
data = res.read()

# Assuming the GeoJSON data is stored in the variable `data`
#geojson = json.loads(data)

# Extract the screenTemperature data from the timeSeries list
#temps = [f['properties']['timeSeries'][i]['screenTemperature'] for i, f in enumerate(geojson['features'])]

# Convert the list to a numpy array
#temps_np = np.array(temps)

#print(temps_np)

print(data.decode("utf-8"))


#MetOffice DataPoint API Key:  9098bf87-f25a-4f96-b4d9-59c8ef8efa2c

#http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/datatype/351923?&key=9098bf87-f25a-4f96-b4d9-59c8ef8efa2c

#http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/datatype/sitelist?&key=9098bf87-f25a-4f96-b4d9-59c8ef8efa2c

#http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/xml/sitelist?&key=9098bf87-f25a-4f96-b4d9-59c8ef8efa2c

#http://datapoint.metoffice.gov.uk/public/data/val/wxobs/all/xml/3351?&key=9098bf87-f25a-4f96-b4d9-59c8ef8efa2c

#http://datapoint.metoffice.gov.uk/public/data/val/wxobs/datatype/3351?&key=9098bf87-f25a-4f96-b4d9-59c8ef8efa2c

#Holmes Chapel location ID = 351923 (elevation 60m)  

#Wilmslow location ID = 354237 (elevation 71m)

#Rostherne location ID = 3351 (elevation 35m)
