# Q4 image provenance

Descriptions below were recorded in sources.json before running either classifier.

## Dog (clear)

A fluffy white dog with pointed ears sits on green grass, with dark foliage behind it. The dog is the clear main subject; its exact breed is not independently established.

File: dog.jpg
Source URL: https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
Source page: https://github.com/pytorch/hub/blob/master/images/dog.jpg
SHA-256: f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a

## Cat (clear)

A close-up of a brown-orange striped cat shows its face, green eyes, whiskers, and part of its chest. The image identifies a domestic cat clearly, but does not establish a breed.

File: cat.png
Source URL: https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/chelsea.png
Source page: https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/chelsea.png
SHA-256: 596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb

## Bus (hard)

A blue-and-white city bus is parked on a sunny street in front of a building. Several pedestrians partially occlude it, and bollards, pavement, signs, and windows add clutter; the bus is the main object of interest.

File: bus.jpg
Source URL: https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg
Source page: https://github.com/ultralytics/assets/blob/main/im/bus.jpg
SHA-256: c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63

## Code references

- Assignment PDF: Q4 usage guide and fixed preprocessing/target protocol.
- Torchvision model/weight API: https://docs.pytorch.org/vision/stable/models.html
- Grad-CAM usage example adapted for the specified ResNet-18 target layer and per-image top class: https://github.com/jacobgil/pytorch-grad-cam#usage-examples
