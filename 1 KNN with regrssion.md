# KNN

### classification vs regression
classification - which bucket will a ball go into
regression - what is the price of a new house

### Data info

long , Latitude  -> house-price
features  labels

### KNN ALgorithm
1- find the distances between features and prediction point
2- Sort from lowest point to greatest
3- Take the top K records
4- Average the label value of those top k records(in classification it would be max label of neighbors)

           
##### Step 1: find the distances between features and prediction point

``` javascript
const features = tf.tensor([
  [-121,47],
  [-121.2,46.5],
  [-122,46.4],
  [-120.9,46.7]
]);

const labels = tf.tensor([
  [200],[250],[215],[240]
  ]);

const predictionPoint = tf.tensor([-121,47]);

// 1) Distances - pythagorous
// features.sub(predictionPoint) // - broadcasting feature substracts 
// [[0 , 0 ], [-0.1999969, -0.5 ], [-1 , -0.5999985], [0.0999985 , -0.2999992]]
// 2) square them with pow(2)
// features.sub(predictionPoint).pow(2)
// [[0 , 0 ], [0.0399988, 0.25 ], [1 , 0.3599982], [0.0099997, 0.0899995]]
// 3) sum each of them
//features.sub(predictionPoint).pow(2).sum(1)
//[0, 0.2899988, 1.3599982, 0.0999992]
// 4) square root each of them

features.sub(predictionPoint).pow(2).sum(1).pow(.5)

//[0, 0.5385153, 1.1661897, 0.3162265]
```
now, we got pythogorous distances 

##### Step2: Sort from lowest point to greatest
- Problem 1: if we sort our features tensor, then order will be messed up with our labels tensor
- Problem 2: Tensors doesnt have sort method.

###### Solution for problem 1 = Concat them now then sort
we cannot do below - because we cannot concat 1D tensor to 2D tensor 
`features.sub(predictionPoint).pow(2).sum(1).pow(0.5).concat(labels)`
one solution is passing true as second argument or use expandDims()
```javascript
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).shape
[4]
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims().shape
[1,4]
// change the dimension by passing to expandDims(1)
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).shape
```
So now lets concat
but this wont work - because default it concats column way so we dont get what we want 
``` javascript 
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels)
//[[0 ], [0.5385153], [1.1661897], [0.3162265], [200 ], [250 ], [215 ], [240 ]]
```
solution is pass second param as '1' to concat
```javascript
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels,1)
//[[0 , 200], [0.5385153, 250], [1.1661897, 215], [0.3162265, 240]]
```
###### Solution for problem 2 = usinhg unstack(splits new tensors on every single ROW and they are all going to be stuffed inside a normal javascript array, then sort that array
```javascript
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels,1).unstack() // this is javascript array of tensors
```
but sort on what - javascript sort needs to know on what it should sort the array of objects
for eg:
```javascript
const distances = [
  {value: 20},
  {value: 30},
  {value: 15},
  {value: 5}
  ];
distances.sort((a,b) => {
  return a.value > b.value ? 1 : -1;
});
//o/p - [{"value":5},{"value":15},{"value":20},{"value":30}]
```
That results us

```javascript
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels,1).unstack().sort((a,b) => {
  return a.get(0)>b.get(0)?1:-1
});
```
##### Step3: Take the top K records
``` javascript
const k =2;
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels,1).unstack().sort((a,b) => {
  return a.get(0)>b.get(0)?1:-1
}).slice(0,k);
// this is javascript array of tensors still you can apply .length to check the length
```
so summing in javascript is needed as we cant apply tensor sum method

```javascript
//finding average using reduce

const distances2 = [
  {value: 20},
  {value: 30},
  {value: 15},
  {value: 5}
  ];

 distances2.reduce((acc, next) => {
  return acc+next.value;
},0) / 4;
//o/p =  17.5
```
so that results

``` javascript
features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels,1).unstack().sort((a,b) => a.get(0)>b.get(0)?1:-1
).slice(0,k).reduce((acc, pair) => (acc+pair.get(1)),0) / k;
//o/p = 220
```


## Practical Nodejs example
```javascript
require('@tensorflow/tfjs-node'); // for CPU - some other calculations might be needed for GPU
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv')

let {features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
});

//console.log(testFeatures);
//console.log(testLabels);

function knn(features, labels, predictionPoint, k){
    //Applying standardization
    const { mean, variance } = tf.moments(features, 0);
    // mean and variance = [47.5600243, -122.2137985]
    // console.log('==============', tf.moments(features, 0)['mean'].print());
    //console.log('--before--', predictionPoint.print());
    // [47.5610008, -122.2259979]
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
    //console.log('--after--', scaledPrediction.print());
    //  [0.0070479, -0.0866189]
    return features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels,1)
        .unstack()
        .sort((a,b) => a.get(0)>b.get(0)?1:-1)
        .slice(0,k)
        .reduce((acc, pair) => (acc+pair.get(1)),0)/k;
}



/*
// For one value
const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);
console.log('Guess--', result, testLabels[0][0]);
//Guess-- 1421200 1085000
// It seems a big variation

//error = (expectedValue-predictedValue) / expectedValue

//const err = (testLabels[0][0]-result)/testLabels[0][0];
//console.log('Error', err*100);
*/
// console.log('Guess--', result, testLabels[0][0]);
//Guess-- 1421200 1085000
// It seems a big variation
/*
error = (expectedValue-predictedValue) / expectedValue
 */
// running out test set on our result

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0]-result)/testLabels[i][0];
    console.log('Error', err*100);
});

/* output:-
    Error -30.98617511520737
    Error -52.95661953727506
    Error -9.552941176470588
    Error -28.528495575221243
    Error -6.069828722002635
    Error -9.855653270993358
    Error -11.176432291666668
    Error 43.34094616639478
    Error -19.536472310319592
    Error -5.603238866396762
 */

// We should consider some other columns as well

// Normalization (or) Standardization
// we use if they are all in similar range no big outliers
// Incase we have outliers we should do standardization which eliminates few outliers
/*
Standardization = (value-average) / Standard Deviation
// note :  stddev = sqrt(variance)
 */
// if we think in tensorflow way , every column we have to calculate separately average and std dev
/* example-----

const numbers = tf.tensor([
  [1,2],
  [3,4],
  [5,6]
 ]);
const {mean, variance} = tf.moments(numbers, 0)
//mean
//variance
// something is wrong we should get column wise

numbers.sub(mean).div(variance.pow(0.5))
 */
// applied standardization in knn function & added 'sqft_lot' column
// after that output looks as:
/*
Error -14.751152073732717
Error -64.06107540702656
Error -99.68823529411765
Error -38.36141592920354
Error -2.9604743083003955
Error -0.3845470293790806
Error -6.091796875
Error 49.44861337683524
Error -18.761893144669433
Error 1.6740890688259111

 */

// Debugging

// node --inspect-brk index.js // run this in console and open chrome
// chrome://inspect/#devices

//In console check ---
//features.shape
//features.print()
//predictionPoint.print()

//hmm debugging looks fine - may be alter value of K or add more features and see
// after added sqft_living
/*
Error -15.323502304147466
Error -11.344580119965723
Error -2.047058823529412
Error 19.327433628318584
Error 7.806324110671936
Error -14.106372465729613
Error -8.782552083333334
Error 13.227406199021207
Error -36.336911441815076
Error 7.381578947368421

 */

```
