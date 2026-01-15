# 9x9 px PNG bitmaps and conifier shapes recognition

## Generating patterns

```console
python3 patterns_maker.py
```

## Generating samples from patterns

```console
python3 patterns_maker.py patterns/ samples/
```

## Training and using the network

```console
pip install numpy pillow scikit-learn matplotlib
```

## Training the network

```console
python3 neural_network.py train samples/
```

## Predict with the neural network

```console
python3 neural_network_system_on_numpy.py predict conifer_model.pkl conifier.png
```
