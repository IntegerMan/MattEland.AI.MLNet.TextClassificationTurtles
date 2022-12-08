using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.Transforms;

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
    new {Utterance="Tonight I dine on turtle soup!", Intent="TurtleFood"},
    new {Utterance="I like turtles!", Intent="TurtlesGood"},
    new {Utterance="Bowser was here!", Intent="Unknown"}
};

IDataView dataView = mlContext.Data.LoadFromEnumerable(phrases);

// Define your training pipeline
Console.WriteLine("Creating Pipeline...");
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Utterance", inputColumnName: "Utterance")
    .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
        labelColumnName: "Intent", sentence1ColumnName: "Utterance"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:"PredictedLabel", inputColumnName:"PredictedLabel"));

// Train the model
Console.WriteLine("Fitting Model...");
TransformerChain<KeyToValueMappingTransformer> model = pipeline.Fit(dataView);

Console.WriteLine("Creating prediction engine");
PredictionEngine<ModelInput, ModelOutput> predictionEngine = 
    mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Generating prediction");
ModelOutput output = predictionEngine.Predict(new ModelInput {Utterance = "Hello World!"});

Console.WriteLine("Predicted intent: " + output.PredictedLabel);

Console.ReadLine();


class ModelInput
{
    public string Utterance { get; set; }
}

class ModelOutput
{
    public string PredictedLabel { get; set; }
}