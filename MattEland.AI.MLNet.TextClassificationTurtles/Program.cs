using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.TorchSharp;

// Source articles / code:
// - https://devblogs.microsoft.com/dotnet/announcing-ml-net-2-0/#sentence-similarity-api
// - https://devblogs.microsoft.com/dotnet/introducing-the-ml-dotnet-text-classification-api-preview/
// - https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis-model-builder
// - https://github.com/dotnet/machinelearning-samples/blob/main/samples/csharp/getting-started/MLNET2/SentenceSimilarity/Program.cs
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

// Create a pipeline for training the model
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"col1", inputColumnName: @"col1")
                        .Append(mlContext.MulticlassClassification.Trainers.TextClassification(labelColumnName: @"col1", sentence1ColumnName: @"col0"))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

// Train the model
Console.WriteLine("Training model...");
var mlModel = pipeline.Fit(dataView);

// Generate a prediction engine
PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

// Generate a series of predictions based on user input
string input;
do
{
    Console.WriteLine("What do you want to say about turtles? (Type Q to Quit)");
    input = Console.ReadLine()!;

    // Get a prediction
    ModelInput sampleData = new(input);
    ModelOutput result = engine.Predict(sampleData);

    // serialize the object to a JSON string
    string json = JsonConvert.SerializeObject(result);

    // Print classification
    Console.WriteLine($"Matched intent {(TurtleIntents)result.PredictedLabel}: {json}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");

Console.WriteLine("Have fun with turtles!");
