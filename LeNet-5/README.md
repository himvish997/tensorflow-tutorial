# Tensorflow Pipline Tutorial on LeNet-5 model
In this tutorial we learn various data pipeline used in the train a model in the tensorflow. there are various method to create the data pipeline in tensorflow by using the `tf.data.Datasets` called iterator. We perform these iterator in the LeNet-5 model of handwritten digit recognitions.

### One shot iterator:
The Dataset can’t be reinitialized once exhausted. To train for more epochs, you would need to repeat the Dataset before feeding to the iterator. This will require huge memory if the size of the data is large. It also doesn’t provide any option to validate the model.<br />
To run:
`python3 main.py --one_shot_iterator`

### Initializable iterator: 
You can dynamically change the Dataset between training and validation Datasets. However, in this case both the Datasets needs to go through the same transformation pipeline.<br />
To run: `python3 main.py --initializable_iterator`

### Re-initializable iterator:
This iterator overcomes the problem of initializable iterator by using two separate Datasets. Each dataset can go through its own preprocessing pipeline. The iterator can be created using the `tf.Iterator.from_structure` method.<br>
To run: `python3 main.py --reinitializable_iterator`

### Feedable iterator:
This iterator provides the option of switching between various iterators. You can create a re-initializable iterator for training and validation purposes. For inference/testing where you require one pass of the dataset, you can use the one shot iterator.<br>
To run: `python3 main.py --feedable_iterator`

#### Reference
1. Blog: https://towardsdatascience.com/building-efficient-data-pipelines-using-tensorflow-8f647f03b4ce <br>
2. Original Source Code: https://github.com/animesh-agarwal/Datasets-and-Iterators/blob/master/DataSet_and_Iterators.ipynb <br>
3. https://www.tensorflow.org/api_docs/python/tf/data/Iterator#from_string_handle
