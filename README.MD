## TAU-F!-Object-Detection

This is a repository for the TAU-F!-Object-Detection project.

## Installation

Clone the repository and install the requirements:

```bash pip install -r requirements.txt ```

## Usage

### Lidar point cloud preprocessing
In order to split the point cloud into frame, InnovizAPI must be installed. To do so

1)Move to lidar preprocessing directory ```cd ./Lidar/preprocessing```

2)Clone InnovizAPI repository

``git clone https://github.com/InnovizTechnologies/InnovizAPI.git``

3) Install the python API

``pip install ./InnovizAPI/python``

4) Run the preprocessing script

``python lidar_record_preprocess.py``