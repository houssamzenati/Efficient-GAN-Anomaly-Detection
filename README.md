# Anomaly-Detection

Anomaly Detection materials, by the Deep Learning 2.0 team in I2R, A*STAR, Singapore

## Prerequisites.
To run the code, follow those steps:

Install Python 3

```
sudo apt install python3 python3-pip
```
Download the project code:

```
git clone https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection
```
Install requirements (in the cloned repository):

```
pip3 install -r requirements.txt
```

## Doing anomaly detection.

Running the code with different options

```
python3 main.py <gan, bigan> <mnist, kdd> run --nb_epochs=<number_epochs> --label=<0, 1, 2, 3, 4, 5, 6, 7, 8, 9> --w=<float between 0 and 1> --m=<'cross-e','fm'> --d=<int> --rd=<int>
```
