using Microsoft.ML.Data;
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
