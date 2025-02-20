# basic-flower-classification
- This simple project will introduce you to the deep learning world based on ***PyTorch***.
- This project completes the flower classification task based on `ResNet34`, whose parameters can be used for transfer learning.



## 1. Data Preparation

1. Get flower dataset from [Flower Classification (kaggle.com)](https://www.kaggle.com/datasets/marquis03/flower-classification).
2. This dataset is well prepared, so that we don't need to worry about coming problems, such as missing or abnormal values. Just read the image path and label from `xxx.csv`. With `PIL.Image`, We can get the image data easily.
3. Use `torchvision.transforms`  to transfer image data to tensor (transfer to the same size, and normalize), and use `torch.utils.data.DataLoader` to encapsulate our image tensor and label.
   - See in `project1-flower-cls.ipynb` to find out how to normalize the image tensor (the value of mean and variance for 3 channels RGB).

## 2. Model Construction

1. Architecture Selection
   - Leverage pre-trained models `ResNet34` for transfer learning: Replace FC layer classification number into our class number (the Kaggle datasets has 14 categories)
2. Or you can define your own model.

## 3. Training Configuration

1. In this section, we define a method `train_valid_model` witch includes the train dataset and validation datasets (for fine-tuning hyper parameters), to train our own model.
   - Keep the best results in every epoch for fear of getting worse result.
   - `model.train()` and `model.eval()` are two different modes. (see in the web)
2. Freeze (**feature extraction**): first only train the FC layer. Parameters from other layers remain unchanged.
3. Unfreeze (**fine-tuned**): then train all the layers. Set `param.required_grad = True for param in model.parameters()`.

## 4. Test

1. Save the models (2 options, see in `project1-flowers-cls.ipynb`) so that next time we can load it to test or resume training.
2. Test image `test.jpg` from the Internet will be classified by forward propagation of the trained model. And what is the most important thing is that the test image should be transfered the same as train/validation datasets (the same size and the same normalization).
3. Result: my model predicted it as `玫瑰`. Unfortunately, it is `郁金香`.



## Conclusion

All the above is the basic process of deep learning. More data preparation methods, evaluation methods and so on are waiting for you to explore!