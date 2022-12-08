using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

// Source articles / code:
// - https://devblogs.microsoft.com/dotnet/announcing-ml-net-2-0/#sentence-similarity-api
// - https://devblogs.microsoft.com/dotnet/introducing-the-ml-dotnet-text-classification-api-preview/
// - https://github.com/dotnet/machinelearning-samples/blob/main/samples/csharp/getting-started/MLNET2/SentenceSimilarity/Program.cs

// Initialize MLContext
MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

// Load your data
Console.WriteLine("Loading Data...");
var phrases = new[]
{
    new {Source="ML.NET 2.0 just released", Target="There's a new version of ML.NET", Similarity=5f},
    new {Source="ML.NET 2.0 just released", Target="The rain in Spain stays mainly in the plain", Similarity=1f}
};

IDataView dataView = mlContext.Data.LoadFromEnumerable(phrases);

// Define your training pipeline
Console.WriteLine("Creating Pipeline...");
SentenceSimilarityTrainer pipeline = mlContext.Regression.Trainers.SentenceSimilarity(
        labelColumnName: "Similarity",
        sentence1ColumnName: "Source",
        sentence2ColumnName: "Target");

// Train the model
Console.WriteLine("Fitting Model...");
NasBertTransformer model = pipeline.Fit(dataView);

Console.WriteLine("Creating prediction engine");
PredictionEngine<ModelInput, ModelOutput> predictionEngine = 
    mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Generating prediction");
ModelOutput output = predictionEngine.Predict(new ModelInput {Source = "Hello World!", Target="Wowzers!"});

Console.WriteLine("Predicted value: " + output.Score);

Console.ReadLine();


class ModelInput
{
    public string Source { get; set; }
    public string Target { get; set; }
}

class ModelOutput
{
    public Single Score { get; set; }
}