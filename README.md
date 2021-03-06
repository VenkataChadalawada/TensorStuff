# TensorStuff
## BASICS:-

### 1 ELEMENT WISE OPERATIONS
###### To create a tensor
```javascript
const data = tf.tensor([1,2,3])
const otherData = tf.tensor([4,5,6])
```
###### shape of tensor
`data.shape`

###### element wise 
```javascript
data.add(otherData)
// 1 2 3] + 4 5 6 = 5 7 9 (1+5  2+5   3+6)
data.sub(otherData)
data.mul(otherData)
data.add(otherData)
```
###### 2D operations
```javascript
const data = tf.tensor([[1,2,3], [4,5,6]]);
const otherData = tf.tensor([[4,5,6], [1,2,3]]);

data.add(otherData) ```
o/p - 
[[5, 7, 9], [5, 7, 9]]
```

### 2 BROAD CASTING OPERATIONS
```javascript
// what if shapes dont match with tensors

const data = tf.tensor([1,2,3]);
const otherData = tf.tensor([5]);

data.add(otherData)
//we still get 
//o/p = [6, 7, 8]
// This is called Broad casting
```

###### 2D broadcasting
```javascript
// eg:1
const data2 = tf.tensor([[1,2,3],[4,5,6]]);
const data = tf.tensor([5,1,6]);
data.add(data2)
//o/p

[[6, 3, 9 ], [9, 6, 12]]

// eg2:
const data2 = tf.tensor([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]);
const data = tf.tensor([5,1,6]);
data.add(data2)
//o/p

[[[6, 3, 9 ], [9, 6, 12]], [[6, 3, 9 ], [9, 6, 12]]]

eg3:
const data1 = tf.tensor([
  [[1,2],[4,5],[7,8]],
  [[1,2],[4,5],[7,8]]
]);
const data2 = tf.tensor([
  [1],
  [1],
  [1]
]);

//SHAPE of data1 is [2,3,2]
//shape of data2 is   [3,1]
// if other tensor shape dimensional value equal to each other or 1 then it is good to smear on
data1.add(data2)
//o/p
[[[2, 3], [5, 6], [8, 9]], [[2, 3], [5, 6], [8, 9]]]

eg4:
const data1 = tf.tensor(
[
  [
    [1,2],
    [4,5],
    [7,8]
  ],
  [[1,2],[4,5],[7,8]]
]);
const data2 = tf.tensor([
  [1],
  [1]
]);

//shape of data1 is [2,3,2]
//shape of data2 is   [2,1]
// if other tensore shape dimensional value equal to each other or 1 then it is good to smear even
// but here last one is 1, but the other one is neither 1 nor equal to other dimensional value so it wont smear
data1.add(data2)
```                            
### 3 LOGGING TENSOR DATA
```javascript
// these tensors are objects
// console.log(data)  would give whole object . 
// but if we just want to see the data
console.log(data.print())
```

### 4 TENSOR ACCESSORS
###### GET
```javascript
const data = tf.tensor([10,20,30]);
data.get(0)
//2D
const data1 = tf.tensor([
  [10,20,30],
  [50,60,80]
]);
data1.get(1,2)
//o/p - 80
data1.get(1,0)
//o/p - 50
```
###### SET -> there is NO SET -> create a new tensor (either by tf.tensor or by some elementry operation resulted to new tensor

### 5 CREATE SLICES
data.slice has two arguments
startindex &  size
eg:
startindex - [0,1] // row , column
size - [6,1] // row, column But, size is not 0 indexed , in here it says 6 rows & 1 column slice

eg:
```javascript
const data1 = tf.tensor([
  [10,20,30],
  [50,60,80],
  [10,20,30],
  [50,60,80],
  [10,20,30],
  [50,60,80],
  [10,20,30],
  [50,60,80],
  [10,20,30],
  [50,60,80]
]);

data1.slice([0,1],[8,1])
/*o/p:
[[20], [60], [20], [60], [20], [60], [20], [60]]*/
```
###### sometimes to specify size you might not know the total number of rows you might end up calculating dynamically with shape
data1.shape
//to access row its in 0th column data1.shape[0]
```javascript
data1.slice([0,1],[data1.shape[0],1]) 
```
###### (or) there is a short cut to get entire row size => "-1"

```javascript data1.slice([0,1],[-1,1]) ```

### 6 TENSOR CONCATENATION
```javascript
const data1 = tf.tensor([
  [10,20,30],
  [50,60,80],
  [10,20,30]
]);
const data2 = tf.tensor([
  [50,60,80],
  [10,20,30],
  [50,60,80]
  ]);

data1.shape
data2.shape
data1.concat(data2)
data1.concat(data2).shape
//But we are looking for [3,6] shape not [6,3] - here concatenation happend vertically
// this is due to Axis 0(column way) 1 (row way)
data1.concat(data2,1).shape // this would give that
```

### 7 SUMMING VALUES ALONG AXIS
```javascript
const jumpData = tf.tensor([
  [10,20,30],
  [50,60,80],
  [10,20,30],
  [50,60,90]
]);
const playerData = tf.tensor([
  [1,160],
  [1,160],
  [1,160],
  [1,160]
  ]);

// Summing values along the axis
// jumpData.sum() // by default it sums up all
jumpData.sum(1) // 10+20+30 here
jumpData.sum(0) //10+50+10+50 here
o/p:-
[60, 190, 60, 200]
[120, 160, 230]
```

### 8 MASSAGING DIMENSIONS
```javascript
const jumpData = tf.tensor([
  [10,20,30],
  [50,60,80],
  [10,20,30],
  [50,60,90]
]);
const playerData = tf.tensor([
  [1,160],
  [1,160],
  [1,160],
  [1,160]
  ]);

// Summing values along the axis
// jumpData.sum() // by default it sums up all
jumpData.sum(1) // 10+20+30 here
jumpData.sum(0) //10+50+10+50 here

//jumpData.sum(1).concat(playerData, 0) //doesnt work
//why
//see the shape of sum result
jumpData.sum(1).shape // 1D [4]
//so we cant concat diff dimension tensors
jumpData.sum(1, true).shape//1D [4,1]
// now lets try
jumpData.sum(1, true).concat(playerData,1)
```
###### EXPAND DIMS
```javascript
// jumpData.sum(1, true).shape //[4,1]
jumpData.sum(1).expandDims().shape //1,4
jumpData.sum(1).expandDims(1).shape //4,1
//solving same problem with expand dims
jumpData.sum(1).expandDims(1).concat(playerData,1)
```

# Linear Regression / Multilinear Regression
- Predicts continuous values which can be constructed with multi dimensional linear fashion
## Steps to be considered
### 1) Getting data into JS world from csv or xls or whatever
``` javascript
let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});
```
### 2) Model Class
##### main Methods
- constructor()
- processfeatures(features)
- train()
- test(testFeatures, testLabels)
- predict(observations)

##### helper Methods
- gradientDescent(features, labels)
- standardize(features)
- recordMSE()
- updateLearningRate()

### 3) Running the Model class
``` javascript
const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
});
// train 
regression.train();

// test - Rsquared (closer to 1 gives better)
const R2 = regression.test(testFeatures, testLabels);
console.log('----R2 is ---------', R2);

// Plot / Data Visualization to understand the model
plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration#',
    yLabel: 'Mean Squared Error'
}); 

// Prediction - for new labels
regression.predict([
    [120, 2, 380],
    [135, 2.1, 420]
]).print();
```

# Logistic Regression
predicts discrete values - classification
Basic Logistic regression
##### 1) Binary classification:
###### Example
- Pass or not
- Spam or not
- customer accepts or customer declines
- apple phone or android phone

Given a persons age, gender , do they prefer read books or watch movies?

we need find a mathematical relationship (FORMULA) between those parameters to predict it
say 
`preference = m1 * Age + m2 * gender + b`


##### 2) Multinomial Logistic Regression



# Binary Logistic Regression

# Multinomial Logistic Regression


# Image Recognition
Given the pixel intensity values in an image identify whether the charecter is a hand-written 0,1,2,3,4,5,6,7,8,9
MNIST DataSet // famous in ML world
Trainig Set - 60,000 Images
Testing Set - 20,000 Images

Every image is collection of pixels
Representing them in greyscale white to black as 0->255



