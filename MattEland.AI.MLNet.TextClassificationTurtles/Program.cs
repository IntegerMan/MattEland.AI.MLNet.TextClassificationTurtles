using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
//using Microsoft.ML.TorchSharp.NasBert;
//using Microsoft.ML.Transforms;

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
var mlModel = pipeline.Fit(dataView);

// Generate a prediction engine
PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

// Generate a series of predictions based on user input
string? input;
do
{
    Console.WriteLine("What do you want to say about turtles? (Type Q to Quit)");
    input = Console.ReadLine();

    // Get a prediction
    ModelInput sampleData = new(input);
    ModelOutput result = engine.Predict(sampleData);

    // Print classification
    Console.WriteLine($"Matched intent {(PossibleOptions)result.PredictedLabel}, {result.Col1}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");

Console.WriteLine("Have fun with the turtles!");

/// <summary>
/// model input class for ReviewSentiment.
/// </summary>
public class ModelInput
{
    public ModelInput(string col0)
    {
        Col0 = col0;
    }

    [LoadColumn(0)]
    [ColumnName(@"col0")]
    public string Col0 { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"col1")]
    public float Col1 { get; set; }

}

/// <summary>
/// model output class for ReviewSentiment.
/// </summary>
public class ModelOutput
{
    [ColumnName(@"col0")]
    public string Col0 { get; set; }

    [ColumnName(@"col1")]
    public uint Col1 { get; set; }

    [ColumnName(@"PredictedLabel")]
    public float PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[] Score { get; set; }

}

public enum PossibleOptions
{
    EatTurtle = 0,
    LikeTurtle = 1,
    Unknown = 2,
    Ninjitsu = 3,
    FastTurtles = 4,
    Recursive = 5,
    TurtleCare = 6,
    TurtleGovernmentalPreferences = 7,
    TurtleIntelligence = 8
}