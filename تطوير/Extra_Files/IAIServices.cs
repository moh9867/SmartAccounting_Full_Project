using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Interfaces
{
    // ===================================================
    // واجهة التنبؤ بالمبيعات
    // ===================================================
    public interface ISalesForecastService
    {
        Task<List<SalesForecastResult>> ForecastNextMonthsAsync(int months = 6);
        Task<SalesForecastResult> ForecastForProductAsync(int productId, DateTime targetDate);
        Task<List<SalesForecastResult>> ForecastByCategoryAsync(int categoryId, int months = 3);
        Task TrainModelAsync();
        Task<double> EvaluateModelAccuracyAsync();
        Task<bool> IsModelTrainedAsync();
    }

    // ===================================================
    // واجهة تحليل مخاطر المخزون
    // ===================================================
    public interface IInventoryRiskService
    {
        Task<List<InventoryRiskResult>> GetAllInventoryRisksAsync();
        Task<List<InventoryRiskResult>> GetHighRiskItemsAsync(float minRiskScore = 0.7f);
        Task<InventoryRiskResult> AnalyzeProductRiskAsync(int productId);
        Task<Dictionary<RiskLevel, int>> GetRiskDistributionAsync();
        Task<decimal> GetTotalAtRiskValueAsync();
        Task TrainModelAsync();
    }

    // ===================================================
    // واجهة التنبؤ بالتدفق النقدي
    // ===================================================
    public interface ICashFlowForecastService
    {
        Task<List<CashFlowForecastResult>> ForecastCashFlowAsync(int months = 6);
        Task<decimal> GetPredictedEndingBalanceAsync(int monthsAhead);
        Task<List<CashFlowForecastResult>> GetNegativeCashFlowPeriodsAsync();
        Task TrainModelAsync();
        Task<double> EvaluateModelAccuracyAsync();
    }

    // ===================================================
    // واجهة تقسيم العملاء
    // ===================================================
    public interface ICustomerSegmentService
    {
        Task<List<CustomerSegmentResult>> SegmentAllCustomersAsync();
        Task<CustomerSegmentResult> SegmentCustomerAsync(int customerId);
        Task<Dictionary<CustomerSegment, List<CustomerSegmentResult>>> GetSegmentedCustomersAsync();
        Task<List<CustomerSegmentResult>> GetAtRiskCustomersAsync();
        Task TrainModelAsync();
    }

    // ===================================================
    // واجهة كشف الشذوذ في المعاملات
    // ===================================================
    public interface IAnomalyDetectionService
    {
        Task<List<TransactionAnomalyResult>> DetectAnomaliesAsync(DateTime? fromDate = null);
        Task<TransactionAnomalyResult> AnalyzeTransactionAsync(int transactionId);
        Task<List<TransactionAnomalyResult>> GetRecentAnomaliesAsync(int count = 10);
        Task TrainModelAsync();
    }

    // ===================================================
    // واجهة لوحة تحكم الذكاء الاصطناعي الموحدة
    // ===================================================
    public interface IAIDashboardService
    {
        Task<AIInsightsDashboard> GetFullDashboardAsync();
        Task<AIHealthMetrics> GetHealthMetricsAsync();
        Task RetrainAllModelsAsync(IProgress<TrainingProgress>? progress = null);
        Task<DateTime?> GetLastTrainingDateAsync();
    }

    // ===================================================
    // واجهة استخراج بيانات التدريب
    // ===================================================
    public interface ITrainingDataService
    {
        Task<List<SalesPredictionInput>> GetSalesTrainingDataAsync();
        Task<List<InventoryRiskInput>> GetInventoryTrainingDataAsync();
        Task<List<CashFlowPredictionInput>> GetCashFlowTrainingDataAsync();
        Task<List<CustomerSegmentInput>> GetCustomerTrainingDataAsync();
        Task<List<AnomalyDetectionInput>> GetTransactionDataForAnomalyAsync();
    }

    // ===================================================
    // نموذج تقدم التدريب
    // ===================================================
    public class TrainingProgress
    {
        public string ModelName { get; set; } = string.Empty;
        public int ProgressPercent { get; set; }
        public string StatusMessage { get; set; } = string.Empty;
        public bool IsCompleted { get; set; }
        public bool HasError { get; set; }
        public string? ErrorMessage { get; set; }
    }
}
