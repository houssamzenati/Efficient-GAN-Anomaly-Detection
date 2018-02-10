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
git clone https://github.com/houssamzenati/Anomaly-Detection.git
```
Install requirements (in the cloned repository):

```
pip3 install -r requirements.txt
```

(Optional) Download the datasets

```
python3 main.py --datasets 
```
(You need an ethernet or VPN connection for this last step!)

Source code based on DL2.0 workshop deep learning

## Doing anomaly detection usings GANs, VAE or AE.

Running the code with different options

```
python3 main.py main.py <gan, ae, vae> <mnist, cifar10> <train, test> <0, 1, 2, 3, 4, 5, 6, 7, 8, 9> <number_epochs>
```
