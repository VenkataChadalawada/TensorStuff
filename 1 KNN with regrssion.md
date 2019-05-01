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
