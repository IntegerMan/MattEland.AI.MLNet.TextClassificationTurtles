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

// (Optional) Use GPU
//mlContext.GpuDeviceId = 0;
//mlContext.FallbackToCpu = false;

var phrases = new[]
{
    new {col0="Tonight I dine on turtle soup!", col1=(float)PossibleOptions.EatTurtle },
    new {col0="I like turtles!", col1=(float)PossibleOptions.LikeTurtle },
};

IDataView dataView = mlContext.Data.LoadFromEnumerable(phrases);

// Data process configuration with pipeline data transformations
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"col1", inputColumnName: @"col1")
                        .Append(mlContext.MulticlassClassification.Trainers.TextClassification(labelColumnName: @"col1", sentence1ColumnName: @"col0"))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

var mlModel = pipeline.Fit(dataView);

PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);


ModelInput sampleData = new(@"Turtles taste good!");
ModelOutput result = engine.Predict(sampleData);

// Print sentiment
Console.WriteLine($"Intent: {(PossibleOptions)result.PredictedLabel} with class labels of {string.Join(", ", result.Score.Select(s => s.ToString("0.00")))}");

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
    EatTurtle,
    LikeTurtle
}