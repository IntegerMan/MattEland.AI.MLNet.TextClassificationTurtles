using Microsoft.ML.Data;

public class ModelOutput
{
    [ColumnName(@"Sentence")]
    public string Sentence { get; set; }

    [ColumnName(@"Label")]
    public uint Label { get; set; }

    [ColumnName(@"PredictedLabel")]
    public float PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[] Score { get; set; }

}
