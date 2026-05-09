using Microsoft.Extensions.Logging;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Services
{
    // ===================================================
    // خدمة لوحة التحكم المركزية للذكاء الاصطناعي
    // ===================================================
    public class AIDashboardService : IAIDashboardService
    {
        private readonly ISalesForecastService _salesForecast;
        private readonly IInventoryRiskService _inventoryRisk;
        private readonly ICashFlowForecastService _cashFlowForecast;
        private readonly ICustomerSegmentService _customerSegment;
        private readonly IAnomalyDetectionService _anomalyDetection;
        private readonly ILogger<AIDashboardService> _logger;
        private readonly string _metadataPath;

        public AIDashboardService(
            ISalesForecastService salesForecast,
            IInventoryRiskService inventoryRisk,
            ICashFlowForecastService cashFlowForecast,
            ICustomerSegmentService customerSegment,
            IAnomalyDetectionService anomalyDetection,
            ILogger<AIDashboardService> logger,
            string modelStoragePath = "MLModels")
        {
            _salesForecast = salesForecast;
            _inventoryRisk = inventoryRisk;
            _cashFlowForecast = cashFlowForecast;
            _customerSegment = customerSegment;
            _anomalyDetection = anomalyDetection;
            _logger = logger;
            _metadataPath = Path.Combine(modelStoragePath, "ai_metadata.json");
        }

        // -----------------------------------------------
        // الحصول على لوحة التحكم الكاملة
        // -----------------------------------------------
        public async Task<AIInsightsDashboard> GetFullDashboardAsync()
        {
            _logger.LogInformation("📊 تجميع لوحة تحكم الذكاء الاصطناعي...");

            var dashboard = new AIInsightsDashboard { LastUpdated = DateTime.Now };

            // تشغيل جميع التحليلات بالتوازي لتسريع الاستجابة
            var salesTask        = _salesForecast.ForecastNextMonthsAsync(6);
            var inventoryTask    = _inventoryRisk.GetHighRiskItemsAsync(0.65f);
            var cashFlowTask     = _cashFlowForecast.ForecastCashFlowAsync(6);
            var customersTask    = _customerSegment.SegmentAllCustomersAsync();
            var anomaliesTask    = _anomalyDetection.GetRecentAnomaliesAsync(10);
            var healthTask       = GetHealthMetricsAsync();

            await Task.WhenAll(salesTask, inventoryTask, cashFlowTask, customersTask, anomaliesTask, healthTask);

            dashboard.SalesForecast      = await salesTask;
            dashboard.HighRiskInventory  = await inventoryTask;
            dashboard.CashFlowForecast   = await cashFlowTask;
            dashboard.CustomerSegments   = await customersTask;
            dashboard.RecentAnomalies    = await anomaliesTask;
            dashboard.HealthMetrics      = await healthTask;

            _logger.LogInformation("✅ لوحة التحكم جاهزة.");
            return dashboard;
        }

        // -----------------------------------------------
        // مقاييس صحة نماذج الذكاء الاصطناعي
        // -----------------------------------------------
        public async Task<AIHealthMetrics> GetHealthMetricsAsync()
        {
            var salesAccuracy     = await _salesForecast.EvaluateModelAccuracyAsync();
            var cashFlowAccuracy  = await _cashFlowForecast.EvaluateModelAccuracyAsync();
            var anomalies         = await _anomalyDetection.GetRecentAnomaliesAsync(100);

            var lastTrainingDate  = await GetLastTrainingDateAsync();
            var daysAgo = lastTrainingDate.HasValue
                ? (int)(DateTime.Now - lastTrainingDate.Value).TotalDays
                : 999;

            return new AIHealthMetrics
            {
                SalesPredictionAccuracy  = salesAccuracy,
                InventoryRiskAccuracy    = 0.85,  // تقدير افتراضي
                CashFlowAccuracy         = cashFlowAccuracy,
                TotalAnomaliesDetected   = anomalies.Count,
                ModelsLastTrainedDaysAgo = daysAgo,
                ModelsNeedRetraining     = daysAgo > 30
            };
        }

        // -----------------------------------------------
        // إعادة تدريب جميع النماذج مع تتبع التقدم
        // -----------------------------------------------
        public async Task RetrainAllModelsAsync(IProgress<TrainingProgress>? progress = null)
        {
            var models = new[]
            {
                ("التنبؤ بالمبيعات",    (Func<Task>)_salesForecast.TrainModelAsync),
                ("مخاطر المخزون",        (Func<Task>)_inventoryRisk.TrainModelAsync),
                ("التدفق النقدي",         (Func<Task>)_cashFlowForecast.TrainModelAsync),
                ("تقسيم العملاء",        (Func<Task>)_customerSegment.TrainModelAsync),
                ("كشف الشذوذ",           (Func<Task>)_anomalyDetection.TrainModelAsync),
            };

            for (int i = 0; i < models.Length; i++)
            {
                var (name, trainFunc) = models[i];

                progress?.Report(new TrainingProgress
                {
                    ModelName       = name,
                    ProgressPercent = (int)((double)i / models.Length * 100),
                    StatusMessage   = $"جاري تدريب نموذج: {name}..."
                });

                try
                {
                    _logger.LogInformation("🔄 تدريب: {Model}", name);
                    await trainFunc();

                    progress?.Report(new TrainingProgress
                    {
                        ModelName       = name,
                        ProgressPercent = (int)((double)(i + 1) / models.Length * 100),
                        StatusMessage   = $"✅ اكتمل تدريب: {name}"
                    });
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ فشل تدريب نموذج {Model}", name);
                    progress?.Report(new TrainingProgress
                    {
                        ModelName    = name,
                        HasError     = true,
                        ErrorMessage = ex.Message,
                        StatusMessage = $"❌ فشل: {name} - {ex.Message}"
                    });
                }
            }

            // حفظ تاريخ آخر تدريب
            await File.WriteAllTextAsync(_metadataPath,
                System.Text.Json.JsonSerializer.Serialize(new { LastTraining = DateTime.Now }));

            progress?.Report(new TrainingProgress
            {
                IsCompleted     = true,
                ProgressPercent = 100,
                StatusMessage   = "✅ اكتمل تدريب جميع النماذج!"
            });
        }

        // -----------------------------------------------
        // تاريخ آخر تدريب
        // -----------------------------------------------
        public async Task<DateTime?> GetLastTrainingDateAsync()
        {
            try
            {
                if (!File.Exists(_metadataPath)) return null;
                var json = await File.ReadAllTextAsync(_metadataPath);
                var metadata = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, DateTime>>(json);
                return metadata?.TryGetValue("LastTraining", out var dt) == true ? dt : null;
            }
            catch
            {
                return null;
            }
        }
    }
}
