const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
class LinearRegression {
    constructor(features, labels, options){
        this.features = this.processfeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);
        this.weights = tf.zeros([this.features.shape[1], 1]); // for m and b
    }
    gradientDescent(){
       // Mean Squared Error = 1/n E ((m xi + b) - Actuali))^2
       // Slope of MSE w.r.t B => d(MSE)/d(b) =  2/nE((m xi+b) - Actuali)
       // Slope of MSE w.r.t M => d(MSE)/dm = 2/nE -xi(Actuali - mxi+bi))
       // where xi is parameter and Actual is the actual label value for that parameter
        const currentGuesses = this.features.matMul(this.weights); // Matrix multiplication in Tensor Flow
        const differences = currentGuesses.sub(this.labels);
        const slopes = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }
    train(){
        for(let i=0; i<this.options.iterations; i++){
            console.log('===',this.options.learningRate);
            this.gradientDescent();
            this.recordMSE();
            this.updateLearningRate();
        }
    }
    test(testFeatures, testLabels){
        testFeatures = this.processfeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        const predictions = testFeatures.matMul(this.weights);
        // predictions.print(); // we get prediction miles per gallon
        const res = testLabels.sub(predictions)
            .pow(2)
            .sum();
        
        const tot = testLabels.sub(testLabels.mean())
                        .pow(2)
                        .sum();
        const rsquare = tf.ones([1,1]).sub(res.div(tot));
        rsquare.print();   
        const rsquareValue = rsquare.dataSync();
        const arr = Array.from(rsquareValue);
        // console.log(rsquareValue);
        console.log('-----rsquareValue----', arr[0]);
                  
    }
    // Data preprocessing - Standarization
    processfeatures(features) {
        features = tf.tensor(features);
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        // we dont need to recalculate mean and variance
        if(this.mean && this.variance){
            // root of variance is standard deviation
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }
        
        return features;
    }

    standardize(features){
        const {mean, variance} = tf.moments(features, 0);
        this.mean = mean;
        this.variance = variance;
        return features.sub(mean).div(variance.pow(0.5));

    }

    recordMSE(){
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0]);
        const mseValue = mse.dataSync();
        const mse_val = Array.from(mseValue);
        // console.log('=========MSE===', MSE);
        this.mseHistory.unshift(mse_val);
    }

    updateLearningRate(){
        if(this.mseHistory.length<2){
            return;
        }
        if(this.mseHistory[0] > this.mseHistory[1]){
            this.options.learningRate/=2;
        } else {
            this.options.learningRate*=1.05;
        }
    }
}


/* // execution
new LinearRegression(features, labels, {
    iterations: 99,
    learningRate: 0.01
})
*/

module.exports = LinearRegression;

/*

JS Array style of slope 
gradientDescent(){
       // Mean Squared Error = 1/n E ((m xi + b) - Actuali))^2
       // Slope of MSE w.r.t B => d(MSE)/d(b) =  2/nE((m xi+b) - Actuali)
       // Slope of MSE w.r.t M => d(MSE)/dm = 2/nE -xi(Actuali - mxi+bi))
       // where xi is parameter and Actual is the actual label value for that parameter
        const currentGuessesForMPG = this.features.map( row => {
            return this.m*row[0] + this.b;
        });

        const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
            return guess - this.labels[i][0];
        })) * 2 / this.features.length;

        const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
            return -1*this.features[i][0] * (this.labels[i][0] - guess);
        }))*2/this.features.length;

        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;

    }


Tensor FLow Matrix(tensor) multiplication basics
4X(2) multiplies (2)X3  = YES we can multiply and we get 4 X 3 result matrix
2X(3) multiplies (4)X2 = NO we cant multiply as 3 and 4 are not same 

// STANDARDIZATION EXAMPLE - Data preprocessing
const features = tf.tensor([
    [10],
    [20],
    [35],
    [95]
]);

const {mean, variance} = tf.moments(features, 0);
// STANDARDIZATION formula (zscores)
features.sub(mean).div(variance.pow(0.5));


Tensorflow uses WebGL for matrices computation

*/

