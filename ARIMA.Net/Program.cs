// See https://aka.ms/new-console-template for more information

using ARIMANet;
using Microsoft.Data.Analysis;
using Microsoft.ML;

var p = 24 * 7;
var q = 24;
var d = 1;

var context = new MLContext();
var dataPath = @"sonar_us_180days.csv";
// use previous 2 weeks data
var df = DataFrame.LoadCsv(dataPath);
var arima = new ARIMA(context, p, q, d);
var X = df["load"].Cast<float>();
var testSize = 100;
var trainX = X.SkipLast(testSize);
var testX = X.TakeLast(testSize);

arima.Fit(trainX);

double RMSE(IEnumerable<float> x, IEnumerable<float> y)
{
    var mse = Enumerable.Zip(x, y).Select(a => (a.First - a.Second) * (a.First - a.Second)).Average();
    return Math.Sqrt(mse);
}

var res = new List<float>();
foreach(var x in testX)
{
    var r = arima.Predict();
    res.Add(r.First());
    arima.Update(x);
}

Console.WriteLine(RMSE(res, testX));

