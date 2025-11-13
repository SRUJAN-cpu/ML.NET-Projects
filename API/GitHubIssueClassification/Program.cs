using Microsoft.ML;

string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

MLContext _mlContext;
PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;

_mlContext = new MLContext(seed: 0);

_trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

//The ProcessData method executes the following tasks:

//Extracts and transforms the data.
//Returns the processing pipeline.

var pipeline = ProcessData();

var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

Evaluate(_trainingDataView.Schema);

PredictIssue();

IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label").Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
.Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized")).Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
.AppendCacheCheckpoint(_mlContext);

    //    Warning
    //Use AppendCacheCheckpoint for small / medium datasets to lower training time.Do NOT use it(remove.AppendCacheCheckpoint()) when handling very large datasets.

    return pipeline;
}

//The BuildAndTrainModel method executes the following tasks:

//Creates the training algorithm class.
//Trains the model.
//Predicts area based on training data.
//Returns the model.

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    _trainedModel = trainingPipeline.Fit(trainingDataView);
    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

    GitHubIssue issue = new GitHubIssue()
    {
        Title = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    var prediction = _predEngine.Predict(issue);
    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
    return trainingPipeline;
}


//The Evaluate method executes the following tasks:

//Loads the test dataset.
//Creates the multiclass evaluator.
//Evaluates the model and create metrics.
//Displays the metrics.


void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
    var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");

    SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
}

void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
}

//The PredictIssue method executes the following tasks:

//Loads the saved model.
//Creates a single issue of test data.
//Predicts area based on test data.
//Combines test data and predictions for reporting.
//Displays the predicted results.

void PredictIssue()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
    GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    var prediction = _predEngine.Predict(singleIssue);
    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}

Console.WriteLine("\nPress any key to exit...");
Console.ReadKey();