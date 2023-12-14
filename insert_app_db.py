import mysql.connector
import json
import os

mydb = mysql.connector.connect(
  host='localhost',
  user='root',
  password=os.getenv('DBPASS'),
  database='FogBus2_Applications',
)

cursor = mydb.cursor()

sql_insert = 'INSERT INTO applications (id, name, tasksWithDependency, entryTasks) VALUES (%s, %s, %s, %s)'

_id = 9
name = 'SmartCCTV'
dep = {
    'CCTVInference': {'parents': ['Sensor'], 'children': ['Actuator']}
}
entry = ['CCTVInference']
cursor.execute(sql_insert, (_id, name, json.dumps(dep), json.dumps(entry)))
# mydb.commit()

cursor.execute("SELECT * FROM applications ORDER BY updatedTime DESC")

res = cursor.fetchone()

print(res)
print(res[1])
dep = json.loads(res[2])
print(dep)
print(json.loads(res[3]))