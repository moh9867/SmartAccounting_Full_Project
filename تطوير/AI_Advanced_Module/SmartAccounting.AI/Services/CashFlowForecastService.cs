using Microsoft.Extensions.Logging;
using Microsoft.ML;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Services
{
    // ===================================================
    // خدمة التنبؤ بالتدفق النقدي
    // ===================================================
    public class CashFlowForecastService : ICashFlowForecastService
    {
        private readonly MLContext _mlContext;
        private readonly ITrainingDataService _trainingDataService;
        private readonly ILogger<CashFlowForecastService> _logger;
        private readonly string _modelPath;

        private ITransformer? _model;
        private PredictionEngine<CashFlowPredictionInput, CashFlowPredictionOutput>? _predictionEngine;
        private bool _isModelLoaded = false;

        public CashFlowForecastService(
            ITrainingDataService trainingDataService,
            ILogger<CashFlowForecastService> logger,
            string modelStoragePath = "MLModels")
        {
            _mlContext = new MLContext(seed: 42);
            _trainingDataService = trainingDataService;
            _logger = logger;
            _modelPath = Path.Combine(modelStoragePath, "cashflow_model.zip");

            Directory.CreateDirectory(modelStoragePath);
            TryLoadSavedModel();
        }

        // -----------------------------------------------
        // تدريب نموذج التدفق النقدي
        // -----------------------------------------------
        public async Task TrainModelAsync()
        {
            _logger.LogInformation("🤖 بدء تدريب نموذج التدفق النقدي...");

            var trainingData = await _trainingDataService.GetCashFlowTrainingDataAsync();
            if (trainingData.Count < 12)
            {
                _logger.LogWarning("⚠️ بيانات غير كافية ({Count} شهر). مطلوب 12 شهراً على الأقل.", trainingData.Count);
                return;
            }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.15);

            var pipeline = _mlContext.Transforms
                .Concatenate("Features",
                    nameof(CashFlowPredictionInput.Month),
                    nameof(CashFlowPredictionInput.Year),
                    nameof(CashFlowPredictionInput.TotalReceivables),
                    nameof(CashFlowPredictionInput.TotalPayables),
                    nameof(CashFlowPredictionInput.PreviousCashBalance),
                    nameof(CashFlowPredictionInput.AverageCollectionDays),
                    nameof(CashFlowPredictionInput.AveragePaymentDays))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.Regression.Trainers.Sdca(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    maximumNumberOfIterations: 100));

            _model = pipeline.Fit(splitData.TrainSet);

            var predictions = _model.Transform(splitData.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            _logger.LogInformation("✅ تدريب التدفق النقدي ناجح | R²={R2:F4} | MAE={MAE:F2}",
                metrics.RSquared, metrics.MeanAbsoluteError);

            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);
            _isModelLoaded = true;

            _predictionEngine?.Dispose();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<CashFlowPredictionInput, CashFlowPredictionOutput>(_model);
        }

        // -----------------------------------------------
        // التنبؤ بالتدفق النقدي للأشهر القادمة
        // -----------------------------------------------
        public async Task<List<CashFlowForecastResult>> ForecastCashFlowAsync(int months = 6)
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var trainingData = await _trainingDataService.GetCashFlowTrainingDataAsync();
            var results = new List<CashFlowForecastResult>();

            var avgReceivables = trainingData.Any() ? trainingData.Average(d => d.TotalReceivables) : 0f;
            var avgPayables = trainingData.Any() ? trainingData.Average(d => d.TotalPayables) : 0f;
            var lastBalance = trainingData.Any() ? trainingData.Last().PreviousCashBalance : 0f;
            var today = DateTime.Today;

            decimal cumulativeCashFlow = (decimal)lastBalance;

            for (int i = 1; i <= months; i++)
            {
                var targetDate = today.AddMonths(i);

                var input = new CashFlowPredictionInput
                {
                    Month = targetDate.Month,
                    Year = targetDate.Year,
                    TotalReceivables = avgReceivables,
                    TotalPayables = avgPayables,
                    PreviousCashBalance = (float)cumulativeCashFlow,
                    AverageCollectionDays = 30f,
                    AveragePaymentDays = 45f
                };

                float predictedNetCashFlow = 0;
                if (_predictionEngine != null)
                {
                    var prediction = _predictionEngine.Predict(input);
                    predictedNetCashFlow = prediction.PredictedCashFlow;
                }
                else
                {
                    predictedNetCashFlow = avgReceivables - avgPayables;
                }

                var netCash = (decimal)predictedNetCashFlow;
                cumulativeCashFlow += netCash;

                var isNegative = netCash < 0;
                string alert = string.Empty;
                if (cumulativeCashFlow < 0)
                    alert = "🚨 تحذير: رصيد سالب متوقع! يجب تأمين تمويل عاجل.";
                else if (netCash < 0)
                    alert = "⚠️ تدفق سلبي هذا الشهر. راجع التزامات الدفع.";

                results.Add(new CashFlowForecastResult
                {
                    ForecastMonth = targetDate,
                    ExpectedInflow = (decimal)avgReceivables,
                    ExpectedOutflow = (decimal)avgPayables,
                    PredictedNetCashFlow = netCash,
                    CumulativeCashFlow = cumulativeCashFlow,
                    IsNegative = isNegative || cumulativeCashFlow < 0,
                    Alert = alert
                });
            }

            return results;
        }

        // -----------------------------------------------
        // الرصيد المتوقع بعد X أشهر
        // -----------------------------------------------
        public async Task<decimal> GetPredictedEndingBalanceAsync(int monthsAhead)
        {
            var forecast = await ForecastCashFlowAsync(monthsAhead);
            return forecast.LastOrDefault()?.CumulativeCashFlow ?? 0;
        }

        // -----------------------------------------------
        // الفترات ذات التدفق السلبي
        // -----------------------------------------------
        public async Task<List<CashFlowForecastResult>> GetNegativeCashFlowPeriodsAsync()
        {
            var forecast = await ForecastCashFlowAsync(12);
            return forecast.Where(f => f.IsNegative).ToList();
        }

        // -----------------------------------------------
        // تقييم دقة النموذج
        // -----------------------------------------------
        public async Task<double> EvaluateModelAccuracyAsync()
        {
            if (!_isModelLoaded) return 0;

            var trainingData = await _trainingDataService.GetCashFlowTrainingDataAsync();
            if (trainingData.Count < 5) return 0;

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var predictions = _model!.Transform(splitData.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            return Math.Max(0, metrics.RSquared);
        }

        private void TryLoadSavedModel()
        {
            try
            {
                if (File.Exists(_modelPath))
                {
                    _model = _mlContext.Model.Load(_modelPath, out _);
                    _predictionEngine = _mlContext.Model.CreatePredictionEngine<CashFlowPredictionInput, CashFlowPredictionOutput>(_model);
                    _isModelLoaded = true;
                    _logger.LogInformation("✅ تم تحميل نموذج التدفق النقدي.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ فشل تحميل نموذج التدفق النقدي: {Message}", ex.Message);
            }
        }
    }
}
