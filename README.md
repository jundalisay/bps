## Behavioral Prediction System for the 3rd PNP ITMS Hackathon

![Splash](bps.jpg)

- Juan Dalisay Jr.
- Ehmil Cataluna
- Francisco Palmares

### Website: https://superphysics.org/bps/

### Google Colab: https://colab.research.google.com/drive/1tqQXIosRe6raZaHaFjAVyP8K0BeFPCE-?usp=sharing


> start time: 9am Oct 18, 2022

1. Import hands dataset from Kaggle and get only right hand palms  https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset/code?resource=download

2. Create Python Script on Google Colab

- import modules
- data
  - upload data to collab  
  - assign to var
- dataset
  - create dataset params
  - split dataset to training 80% (train_ds) and validation 20% (val_ds)
  - find or set class names
  - pass the dataset to the Keras Model.fit method for training 
  - config dataset for performance
  - standardize the data in the dataset by normalizing
- model
  - create and compile the model
  - get summary
- train
- visualize results
- test on samples
  - Hand_0005083.jpg
  - Hand_0005116.jpg


> end time: 12 midnight Oct 19, 2022 (15 hours)




### Problem/s Encountered

- Limited samples causes overfitting
- Palm images too varied

### Workaround

- Data augmentation -- generate additional training data from existing examples
- Manually select palm images that are proper
