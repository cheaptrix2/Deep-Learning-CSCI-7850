# Deep Learning CSCI 7850

## PyTorch
### Dataset
The source of the dataset is below: <br>
```https://www.kaggle.com/datasets/devavratatripathy/ecg-dataset```
#### Downloading the dataset
1. Open a web browser.
2. Input the URL below into the URL bar in the web browser<br>
* ```https://storage.cloud.google.com/deep-learning-csci-7850/ecg.csv```
3. This will download the dataset to your local machine's folder designated for downloads.

### Model Weights
The model weights and logs can be located at the below Google Cloud Bucket link:<br>
```https://storage.cloud.google.com/deep-learning-csci-7850/epoch%3D99-step%3D11000.ckpt```

### Running the model
#### Required Software
1. Ensure you have the following downloaded and installed in the environment you will be running the program:<br>
* python3
* JuypterNotebook
2. Navigate to the directory that you want to run the model in.
3. Move the dataset file into this directory.
4. Download the JuypterNotebook named ```run.ipynb```.
5. Open notebook and run all cells.
6. View results.

#### Run as Script
Alternately, download run.py and run the file from a terminal on the command line as such:
1. Repeat steps 1 - 3 above.
2. Download the python file ```run.py```
3. Execute the command ```$python run.py```
4. View results.

### Source Files
```Term_Project.ipynb``` and ```Term_Project.py``` are source files that were run for this experiment.<br>
It is suggested to use ```run.ipynb``` or ```run.py``` to view results.

# PySpark

## Starting and running application

### Installing Required Software
* Install PySpark
  * ```$pip install pyspark```
* Install FindSpark:
  * ```$pip install findspark```

### Running the Models JupyterNotebook
* Download ```Project_Serial.ipynb```
  * Run all cells and view results.

### Running the Models on command line
* Download ```Project_Serial.py```
* Run on command line with ```$python3 ./Project_Serial.py```
* View results
  * It is suggested to redirect the file's output to a new file for readability.
