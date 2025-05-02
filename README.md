# Sign Language Interpreter

## Description
This project is a real-time Sign Language Interpreter that uses computer vision and machine learning to recognize American Sign Language (ASL) letters. By using your computer's webcam, the system detects hand gestures and predicts which ASL letter you are signing. The model is trained on a custom dataset and can recognize 24 letters of the ASL alphabet, excluding "J" and "Z" (due to their dynamic nature). The trained model is saved for future use, providing a seamless way to interpret ASL signs.

## Files 
- `README.md`: Provides guidance and instructions on how to use the Sign Language Interpreter, including setup, steps, and examples.
- `collect_imgs.py`: Collects and saves images of ASL gestures from the webcam to create a dataset for training the classifier.
- `create_dataset.py`: Processes collected hand gesture images and extracts features (hand landmarks) to create a dataset for model training.
- `train_classifier`: Trains a machine learning classifier (Random Forest) on the dataset to predict ASL hand gestures.
- `inference_classifier.py`: Runs real-time inference using the trained model to recognize and display predicted ASL letters from webcam input.

## Citation

### Data
  - The ISS data comes directly from NASA's website which is the most updated and recent data regarding the International Space Station. The data is presented as a .txt and .XML, for this script, we will be using the .XML:
[     https://spotthestation.nasa.gov/trajectory_data.cfm 
](https://nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.xml)https://nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.xml

Click the link above to download the data and save it to the same folder as your python script.

## How to Run

<details>
<summary>Containerized App</summary>
- To begin, head over to the directory that contains both the Dockerfile and the docker-compose.yml files.
- Build the Docker image using the command:
  
  `docker build -t username/flask-app:1.0 .`

  Replace username and flask-app with your own respective information.

- Then run the following Docker command which will start the container and run the Flask app inside of it.
     
  `docker-compose up`
</details>

<details>
<summary>Unit Test</summary>
- First, after check to see if any things are up and running with:

```docker ps -a```

  - Close all servers with:
    
    ``` docker stop <Container ID> ```
  
  - Next, start the service in the background with the command:

    ``` docker-compose up -d ```
  
- Once that is up and running, navigate to the directory with the Unit test (test_iss_tracker.py) and the iss_tracker.py files.
- Run the command:

  ``` pytest -v ```

  - This should run all the unit tests associated with this flask app.
</details>
<details>
<summary>Routes and Outputs</summary>

### Once your Flask app is running properly, you can curl several endpoints using the command:


`curl 127.0.0.1:5000/comment`
 - **Output:** This endpoint is a 'GET' method that will return the 'comment' dictionary associated with the ISS .XML file.

`curl 127.0.0.1:5000/header` 
- **Output:** This endpoint is a 'GET' method that will return the 'header' dictionary associated with the ISS .XML file.

`curl 127.0.0.1:5000/metadata` 
- **Output:** This endpoint is a 'GET' method that will return the 'metadata' dictionary associated with the ISS .XML file.

`curl 127.0.0.1:5000/epochs` 
- **Output:** This endpoint is a 'GET' method that will return the entire data set of epochs associated with the ISS in the .XML file.

`curl 127.0.0.1:5000/epochs/epochs?limit=int&offset=int` 
- **Output:** This endpoint is a 'GET' method that includes query parameters limit and offset are included in the request with their respective integer values. This endpoint is designed to retrieve a list of ISS state vectors with optional pagination using the limit and offset parameters.
- **limit**: The number of epoch records to return.
- **offset**: The starting position in the data set.

`curl 127.0.0.1:5000/epochs/<epoch>`
- **Output:** This endpoint is a 'GET' method that will return the state vectors for a specific epoch from the data set.
- The epoch should be an exact epoch from the data, such as `curl 127.0.0.1:5000/epochs/2024-075T13:56:00.000Z`

`curl 127.0.0.1:5000/epochs/<epoch>/speed`
- **Output:** This endpoint is a 'GET' method that will return the instantaneous speed for a specific epoch from the data set.
- The epoch should be an exact epoch from the data, such as `curl 127.0.0.1:5000/epochs/2024-075T13:56:00.000Z/speed`

`curl 127.0.0.1:5000/epochs/<epoch>/location`
- **Output:** This endpoint is a 'GET' method that will return the location data (latitude, longitude, altitude, and geoposition) for a specific epoch from the data set.
- The epoch should be an exact epoch from the data, such as `curl 127.0.0.1:5000/epochs/2024-075T13:56:00.000Z/location`

`curl 127.0.0.1:5000/now`
- **Output:** This endpoint is a 'GET' method that will return instantaneous speed, latitude, longitude, altitude, and geoposition for the epoch that is nearest in time to when the program is run.

</details>

<details>
<summary>Example Curl</summary>
  
`curl 127.0.0.1:5000/comment`

```
["Units are in kg and m^2",
"MASS=459154.20",
"DRAG_AREA=1487.80",
"DRAG_COEFF=2.00",
"SOLAR_RAD_AREA=0.00",
"SOLAR_RAD_COEFF=0.00",
"Orbits start at the ascending node epoch",
"ISS first asc. node: EPOCH = 2024-03-15T13:05:34.170 $ ORBIT = 402 $ LAN(DEG) = 49.49781",
"ISS last asc. node : EPOCH = 2024-03-30T10:42:10.141 $ ORBIT = 633 $ LAN(DEG) = -3.07552",
"Begin sequence of events","TRAJECTORY EVENT SUMMARY:",null,
"|       EVENT        |       TIG        | ORB |   DV    |   HA    |   HP  |",
"|                    |       GMT        |     |   M/S   |   KM    |   KM    |",
"|                    |                  |     |  (F/S)  |  (NM)   |  (NM)   |",
"=============================================================================",
"71S Launch            081:13:21:19.000             0.0     425.0     412.5",
"(0.0)   (229.5)   (222.8)",null,"71S Docking           081:16:39:42.000             0.0     425.0     412.5",
"(0.0)   (229.5)   (222.7)",
null,"SpX-30 Launch         081:20:55:09.000             0.0     425.0     412.6","(0.0)   (229.5)   (222.8)",
null,"SpX-30 Docking        083:11:30:00.000             0.0     425.3     412.0","(0.0)   (229.6)   (222.4)",
null,"============================================================================="
,"End sequence of events"]
```

`curl 127.0.0.1:5000/header`

```
{"CREATION_DATE":"2024-075T20:59:30.931Z",
"ORIGINATOR":"JSC"}
```

`curl 127.0.0.1:5000/metadata`

```
{
  "CENTER_NAME":"EARTH",
  "OBJECT_ID":"1998-067-A",
  "OBJECT_NAME":"ISS",
  "REF_FRAME":"EME2000",
  "START_TIME":"2024-075T12:00:00.000Z",
  "STOP_TIME":"2024-090T12:00:00.000Z",
  "TIME_SYSTEM":"UTC"
}
```

`curl 127.0.0.1:5000/epochs`

```
....
{
    "EPOCH": "2024-090T10:50:00.000Z",
    "X": 6211.00796878096,
    "X_DOT": -2.71881392442844,
    "Y": 593.261757779672,
    "Y_DOT": 4.94514472792443,
    "Z": 2679.1792595489,
    "Z_DOT": 5.19004672600028
  },
  {
    "EPOCH": "2024-090T10:54:00.000Z",
    "X": 5340.04172598131,
    "X_DOT": -4.49467498035985,
    "Y": 1744.01482432719,
    "Y_DOT": 4.58581135053425,
    "Z": 3811.63886071994,
    "Z_DOT": 4.18928581945629
  },
  {
    "EPOCH": "2024-090T10:58:00.000Z",
    "X": 4079.90977424504,
    "X_DOT": -5.94202859515296,
    "Y": 2767.67650101493,
    "Y_DOT": 3.89261089629353,
    "Z": 4665.53989721581,
    "Z_DOT": 2.88305607823398
  }...
```

`curl 127.0.0.1:5000/epochs/2024-090T10:58:00.000Z`

```
  {
    "EPOCH": "2024-090T10:58:00.000Z",
    "X": 4079.90977424504,
    "X_DOT": -5.94202859515296,
    "Y": 2767.67650101493,
    "Y_DOT": 3.89261089629353,
    "Z": 4665.53989721581,
    "Z_DOT": 2.88305607823398
  }
```

`curl 127.0.0.1:5000/epochs/2024-090T10:58:00.000Z/speed`

```
"7.666298700533425"
```

`curl 127.0.0.1:5000/epochs/2024-090T10:58:00.000Z/location`

```
{
  "Altitude":416.6976987197977,
  "EPOCH":"2024-090T10:58:00.000Z",
  "Geoposition":"Location currently unavailable",
  "Latitude":43.42081705210838,
  "Longitude":49.65167711498843
}
```
- If Geopositon shows "Location currently unavailable" this means the epoch  Latitude and Longitude are over a body of water.


`curl 127.0.0.1:5000/now`

```
{
  "Altitude":428.22771790667775,
  "EPOCH":"2024-078T19:48:00.000Z",
  "Geoposition":"Location currently unavailable",
  "Latitude":-42.81339797977896,
  "Longitude":-120.94411834789345,
  "Speed":"7.654282674187185",
  "X":4975.90962226887,
  "X_DOT":3.22096430613272,
  "Y":-343.072998653738,
  "Y_DOT":6.25756433003512,
  "Z":-4620.84857737651,
  "Z_DOT":3.00937216881748
}
```
- If Geopositon shows "Location currently unavailable" this means the current Latitude and Longitude of the ISS are over a body of water.

  
</details>


<img width="1121" alt="diagram" src="https://github.com/ArshDauwa/ISS-Tracker/assets/127358497/e8a70d80-0c33-4312-bb59-54f58f26f4fa">

The software diagram illustrates the architecture of the project. It consists of three main components: the NASA data API, the Flask app, and the unit testing module. The NASA data API serves as the source of data, providing information about the International Space Station's trajectory. This data is accessed by the Flask app through API calls. The Flask app contains various routes that handle different requests, such as querying specific epochs or retrieving location information. Additionally, the Flask app is subjected to unit testing, ensuring the accuracy and reliability of its functionality. Finally, the unit testing module evaluates the Flask app's performance, providing feedback on whether the tests pass or fail.














