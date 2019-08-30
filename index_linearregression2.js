require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('.././load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
});

// if batch size is 1 then this batch gradient descent becomes stochaistic gradient descent

regression.features.print();

regression.train();
const R2 = regression.test(testFeatures, testLabels);

// console.log('-----MSE HISTORY----', regression.mseHistory);
// console.log('-----B HISTORY----', regression.bHistory);

console.log('----R2 is ---------', R2);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration#',
    yLabel: 'Mean Squared Error'
}); 

/*
plot({
    x: regression.bHistory,
    y: regression.mseHistory.reverse(),
    xLabel: 'Iteration#',
    yLabel: 'Mean Squared Error'
});
*/

// Predicting several inputs at a time
// order is important 
regression.predict([
    [120, 2, 380],
    [135, 2.1, 420]
]).print();


// console.log('Updated weights m , b is:', regression.weights.print());
// with learningRate: 0.001, - -9 NOT GOOD AT ALL
// with learningRate: 0.1, - ~60%
// with learningRate: 0.5, - ~60%
// with learningRate: 1, - ~60%
// probably that is the most it can get with this parameter and learning rate we provided
// we can try different others

// after adding weight and displacement
// 1 -> -Infinity
// 0.01 -> -0.89
// 0.1 -> 0.66
// 0.5 -> 0.65
// may be time to try the iterations
// learningRate 0.1  & iterations 1000 -> 0.66

// Learning Rate Optimization
/*
  there are few methods 
  1) Adam
  2) Adagrad
  3) RMSProp
  4) Momentum
  or Custom

  */
