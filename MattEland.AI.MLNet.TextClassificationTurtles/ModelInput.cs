using Microsoft.ML.Data;
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
