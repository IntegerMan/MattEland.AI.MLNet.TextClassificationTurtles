using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.Transforms;

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

// Load your data
Console.WriteLine("Loading Data...");
IEnumerable<TrainingRow> phrases = new[]
{
    new TrainingRow {Utterance="Tonight I dine on turtle soup!", Intent=(uint) TurtleIntents.EatTurtles},
    new TrainingRow {Utterance="I like turtles!", Intent=(uint)TurtleIntents.TurtlesAreGood},
    new TrainingRow {Utterance="Eat and get out!", Intent=(uint)TurtleIntents.Unknown}
};

IDataView dataView = mlContext.Data.LoadFromEnumerable(phrases);

// Define your training pipeline
Console.WriteLine("Creating Pipeline...");
var pipeline = mlContext.MulticlassClassification.Trainers.TextClassification(
    labelColumnName: @"Intent",
    sentence1ColumnName: @"Utterance");

// Train the model
Console.WriteLine("Fitting Model...");
var model = pipeline.Fit(dataView);

Console.WriteLine("Creating prediction engine");
PredictionEngine<ModelInput, ModelOutput> predictionEngine =
    mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Generating prediction");
ModelOutput output = predictionEngine.Predict(new ModelInput { Col0 = "Hello World!" });

Console.WriteLine("Predicted intent: " + output.PredictedLabel);

Console.ReadLine();

public class TrainingRow
{
    [LoadColumn(0)]
    [ColumnName(@"Utterance")]
    public string Utterance { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Intent")]
    public uint Intent { get; set; }
}

public class ModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"col0")]
    public string Col0 { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"col1")]
    public float Col1 { get; set; }

}

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

public enum TurtleIntents
{
    EatTurtles,
    TurtlesAreGood,
    Unknown
}