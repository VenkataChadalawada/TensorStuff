# KNN

### classification vs regression
classification - which bucket will a ball go into
regression - what is the price of a new house

### KNN ALgorithm
1- find the distances between features and prediction point
2- Sort from lowest point to greatest
3- Take the top K records
4- Average the label value of those top k records(in classification it would be max label of neighbors)

### Data info

long , Latitude  -> house-price
features             labels

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

// Distances - pythagorous
features.sub(predictionPoint).pow(2).sum(1).pow(.5)
```
now
