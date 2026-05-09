using Microsoft.ML.Data;

namespace SmartAccounting.AI.Models
{
    // ===== نماذج التنبؤ بالمبيعات =====
    public class SalesPredictionInput
    {
        [LoadColumn(0)] public float Month { get; set; }
        [LoadColumn(1)] public float Year { get; set; }
        [LoadColumn(2)] public float DayOfWeek { get; set; }
        [LoadColumn(3)] public float ProductId { get; set; }
        [LoadColumn(4)] public float CategoryId { get; set; }
        [LoadColumn(5)] public float PreviousMonthSales { get; set; }
        [LoadColumn(6)] public float AveragePrice { get; set; }
        [LoadColumn(7)] public float CustomerCount { get; set; }
        [LoadColumn(8), ColumnName("Label")] public float ActualSales { get; set; }
    }

    public class SalesPredictionOutput
    {
        [ColumnName("Score")]
        public float PredictedSales { get; set; }
    }

    // ===== نماذج التنبؤ بالمخزون الراكد =====
    public class InventoryRiskInput
    {
        [LoadColumn(0)] public float ProductId { get; set; }
        [LoadColumn(1)] public float DaysSinceLastSale { get; set; }
        [LoadColumn(2)] public float CurrentStock { get; set; }
        [LoadColumn(3)] public float AverageMonthlySales { get; set; }
        [LoadColumn(4)] public float PriceChangePercent { get; set; }
        [LoadColumn(5)] public float SeasonalityIndex { get; set; }
        [LoadColumn(6), ColumnName("Label")] public bool IsSlowMoving { get; set; }
    }

    public class InventoryRiskOutput
    {
        [ColumnName("PredictedLabel")] public bool IsSlowMoving { get; set; }
        [ColumnName("Probability")] public float RiskProbability { get; set; }
        [ColumnName("Score")] public float Score { get; set; }
    }

    // ===== نماذج التنبؤ بالتدفق النقدي =====
    public class CashFlowPredictionInput
    {
        [LoadColumn(0)] public float Month { get; set; }
        [LoadColumn(1)] public float Year { get; set; }
        [LoadColumn(2)] public float TotalReceivables { get; set; }
        [LoadColumn(3)] public float TotalPayables { get; set; }
        [LoadColumn(4)] public float PreviousCashBalance { get; set; }
        [LoadColumn(5)] public float AverageCollectionDays { get; set; }
        [LoadColumn(6)] public float AveragePaymentDays { get; set; }
        [LoadColumn(7), ColumnName("Label")] public float NetCashFlow { get; set; }
    }

    public class CashFlowPredictionOutput
    {
        [ColumnName("Score")]
        public float PredictedCashFlow { get; set; }
    }

    // ===== نموذج تصنيف العملاء =====
    public class CustomerSegmentInput
    {
        [LoadColumn(0)] public float TotalPurchases { get; set; }
        [LoadColumn(1)] public float PurchaseFrequency { get; set; }
        [LoadColumn(2)] public float DaysSinceLastPurchase { get; set; }
        [LoadColumn(3)] public float AverageOrderValue { get; set; }
        [LoadColumn(4)] public float PaymentOnTimeRate { get; set; }
        [LoadColumn(5), ColumnName("Label")] public uint Segment { get; set; }
    }

    public class CustomerSegmentOutput
    {
        [ColumnName("PredictedLabel")] public uint PredictedSegment { get; set; }
        [ColumnName("Score")] public float[] Scores { get; set; } = Array.Empty<float>();
    }

    // ===== نتائج تحليل الشذوذ =====
    public class AnomalyDetectionInput
    {
        [LoadColumn(0), ColumnName("Value")] public float TransactionAmount { get; set; }
    }

    public class AnomalyDetectionOutput
    {
        [ColumnName("Prediction")]
        public double[] Prediction { get; set; } = Array.Empty<double>();
        public bool IsAnomaly => Prediction.Length > 0 && Prediction[0] == 1;
        public double AnomalyScore => Prediction.Length > 1 ? Prediction[1] : 0;
        public double ExpectedValue => Prediction.Length > 2 ? Prediction[2] : 0;
    }

    // ===== DTOs للنتائج المعروضة =====
    public class SalesForecastResult
    {
        public DateTime ForecastDate { get; set; }
        public decimal PredictedAmount { get; set; }
        public decimal LowerBound { get; set; }
        public decimal UpperBound { get; set; }
        public double ConfidenceLevel { get; set; }
        public string Period { get; set; } = string.Empty;
    }

    public class InventoryRiskResult
    {
        public int ProductId { get; set; }
        public string ProductName { get; set; } = string.Empty;
        public string ProductCode { get; set; } = string.Empty;
        public decimal CurrentStock { get; set; }
        public decimal CurrentValue { get; set; }
        public float RiskScore { get; set; }
        public RiskLevel RiskLevel { get; set; }
        public int DaysSinceLastSale { get; set; }
        public decimal AverageMonthlySales { get; set; }
        public int EstimatedMonthsToDeplete { get; set; }
        public string Recommendation { get; set; } = string.Empty;
    }

    public class CashFlowForecastResult
    {
        public DateTime ForecastMonth { get; set; }
        public decimal ExpectedInflow { get; set; }
        public decimal ExpectedOutflow { get; set; }
        public decimal PredictedNetCashFlow { get; set; }
        public decimal CumulativeCashFlow { get; set; }
        public bool IsNegative { get; set; }
        public string Alert { get; set; } = string.Empty;
    }

    public class CustomerSegmentResult
    {
        public int CustomerId { get; set; }
        public string CustomerName { get; set; } = string.Empty;
        public CustomerSegment Segment { get; set; }
        public string SegmentLabel { get; set; } = string.Empty;
        public decimal TotalPurchases { get; set; }
        public int PurchaseFrequency { get; set; }
        public int DaysSinceLastPurchase { get; set; }
        public string RecommendedAction { get; set; } = string.Empty;
    }

    public class TransactionAnomalyResult
    {
        public int TransactionId { get; set; }
        public DateTime TransactionDate { get; set; }
        public string Description { get; set; } = string.Empty;
        public decimal Amount { get; set; }
        public bool IsAnomaly { get; set; }
        public double AnomalyScore { get; set; }
        public double ExpectedAmount { get; set; }
        public string AnomalyReason { get; set; } = string.Empty;
    }

    public class AIInsightsDashboard
    {
        public List<SalesForecastResult> SalesForecast { get; set; } = new();
        public List<InventoryRiskResult> HighRiskInventory { get; set; } = new();
        public List<CashFlowForecastResult> CashFlowForecast { get; set; } = new();
        public List<CustomerSegmentResult> CustomerSegments { get; set; } = new();
        public List<TransactionAnomalyResult> RecentAnomalies { get; set; } = new();
        public AIHealthMetrics HealthMetrics { get; set; } = new();
        public DateTime LastUpdated { get; set; } = DateTime.Now;
    }

    public class AIHealthMetrics
    {
        public double SalesPredictionAccuracy { get; set; }
        public double InventoryRiskAccuracy { get; set; }
        public double CashFlowAccuracy { get; set; }
        public int TotalAnomaliesDetected { get; set; }
        public int ModelsLastTrainedDaysAgo { get; set; }
        public bool ModelsNeedRetraining { get; set; }
    }

    public enum RiskLevel { Low, Medium, High, Critical }
    public enum CustomerSegment { VIP = 1, Regular = 2, AtRisk = 3, Lost = 4 }
}
