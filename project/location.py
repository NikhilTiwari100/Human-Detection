import requests
import json


key = "AIzaSyCXdoEdidWlAydjdavP7CLP65yhU_8wraI"
origin_lat =28.5141047
origin_long= 77.0533326
destination_lat=28.4912229
destination_long  = 77.0119686

data = requests.get("https://maps.googleapis.com/maps/api/distancematrix/json?origins="+str(origin_lat)+","+str(origin_long)+"&destinations="+str(destination_lat)+","+str(destination_long)+"&key="+str(key))

print(data.json())

json_str = json.dumps(data.json())

resp = json.loads(json_str)

shortest_distance = resp['rows'][0]['elements'][0]['distance']['text']
print(shortest_distance)