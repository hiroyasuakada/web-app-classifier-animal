Name
====

animalai

# Description

Web application for Image Recognition of animals using Keras

keyword: Python, TensorFlow, keras, cnn

# Demo

# Dependency

python 3.7.3
numpy 1.16.4
sklearn 0.21.2
tensorflow 1.13.1


# Usage

## 1. Collecting data by cloning (FlickerAPI + urlretrieve)

download.py

## 2. Converting the collected data values to numpy file

gen_data.py: normalizing numbers

gen_data_augmented.py: adding rotated and reversed data to original ones

## 3. Processing learning phase on cnn, transfer learning

animal_cnn.py: learning train data based on gen_data.py

animal_cnn_augmented.py: learning train data based on gen_data_augmented.py

## 4. Showing the result of learning on commandprompt

predict.py

# References

<https://www.udemy.com/tensorflow-advanced/>


