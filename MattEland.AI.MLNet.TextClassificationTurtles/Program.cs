using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

// Reference articles & code:
// - https://devblogs.microsoft.com/dotnet/announcing-ml-net-2-0/#text-classification-scenario-in-model-builder
// - https://devblogs.microsoft.com/dotnet/introducing-the-ml-dotnet-text-classification-api-preview/
// - https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis-model-builder
// - https://github.com/dotnet/machinelearning-samples/blob/main/samples/csharp/getting-started/MLNET2/TextClassification/ReviewSentiment.training.cs

// Initialize MLContext
MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

// Load the data source
Console.WriteLine("Loading data...");
IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    "Turtles.tsv",
    separatorChar: '\t',
    hasHeader: false
);

/** MODEL TRAINING ****************************************************************************/

// To evaluate the effectiveness of machine learning models we split them into a training set for fitting
// and a testing set to evaluate that trained model against unknown data
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 1234);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;

// Create a pipeline for training the model
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                            outputColumnName: "Label", 
                            inputColumnName: "Label")
                        .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
                            labelColumnName: "Label",
                            sentence1ColumnName: "Sentence",
                            architecture: BertArchitecture.Roberta))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                            outputColumnName: "PredictedLabel", 
                            inputColumnName: "PredictedLabel"));

// Train the model using the pipeline
Console.WriteLine("Training model...");
ITransformer model = pipeline.Fit(trainData);

/** MODEL EVALUATION **************************************************************************/

// Evaluate the model's performance against the TEST data set
Console.WriteLine("Evaluating model performance...");

// We need to apply the same transformations to our test set so it can be evaluated via the resulting model
IDataView transformedTest = model.Transform(testData);
MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

// Display Metrics
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
Console.WriteLine();

// Confusion Matrix with class list
Console.WriteLine("Classes:");
foreach (TurtleIntents value in Enum.GetValues<TurtleIntents>())
{
    Console.WriteLine($"{((int)value)}: {value}");
}
Console.WriteLine();

Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

/** PREDICTION GENERATION *********************************************************************/    

// Generate a prediction engine
Console.WriteLine("Creating prediction engine...");
PredictionEngine<ModelInput, ModelOutput> engine = 
    mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Ready to generate predictions.");

// Generate a series of predictions based on user input
string input;
do
{
    Console.WriteLine();
    Console.WriteLine("What do you want to say about turtles? (Type Q to Quit)");
    input = Console.ReadLine()!;

    // Get a prediction
    ModelInput sampleData = new(input);
    ModelOutput result = engine.Predict(sampleData);

    // Print classification
    float maxScore = result.Score[(uint)result.PredictedLabel];
    Console.WriteLine($"Matched intent {(TurtleIntents)result.PredictedLabel} with score of {maxScore:f2}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");

Console.WriteLine("Have fun with turtles!");
