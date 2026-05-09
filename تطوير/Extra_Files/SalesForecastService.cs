using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Services
{
    // ===================================================
    // خدمة التنبؤ بالمبيعات باستخدام ML.NET
    // ===================================================
    public class SalesForecastService : ISalesForecastService
    {
        private readonly MLContext _mlContext;
        private readonly ITrainingDataService _trainingDataService;
        private readonly ILogger<SalesForecastService> _logger;
        private readonly string _modelPath;

        private ITransformer? _model;
        private PredictionEngine<SalesPredictionInput, SalesPredictionOutput>? _predictionEngine;
        private bool _isModelLoaded = false;

        public SalesForecastService(
            ITrainingDataService trainingDataService,
            ILogger<SalesForecastService> logger,
            string modelStoragePath = "MLModels")
        {
            _mlContext = new MLContext(seed: 42);
            _trainingDataService = trainingDataService;
            _logger = logger;
            _modelPath = Path.Combine(modelStoragePath, "sales_forecast_model.zip");

            Directory.CreateDirectory(modelStoragePath);
            TryLoadSavedModel();
        }

        // -----------------------------------------------
        // تدريب نموذج التنبؤ بالمبيعات
        // -----------------------------------------------
        public async Task TrainModelAsync()
        {
            _logger.LogInformation("🤖 بدء تدريب نموذج التنبؤ بالمبيعات...");

            var trainingData = await _trainingDataService.GetSalesTrainingDataAsync();
            if (trainingData.Count < 50)
            {
                _logger.LogWarning("⚠️ بيانات التدريب غير كافية ({Count} سجل). مطلوب 50 على الأقل.", trainingData.Count);
                return;
            }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // بناء Pipeline التدريب
            var pipeline = _mlContext.Transforms
                .Concatenate("Features",
                    nameof(SalesPredictionInput.Month),
                    nameof(SalesPredictionInput.Year),
                    nameof(SalesPredictionInput.DayOfWeek),
                    nameof(SalesPredictionInput.ProductId),
                    nameof(SalesPredictionInput.CategoryId),
                    nameof(SalesPredictionInput.PreviousMonthSales),
                    nameof(SalesPredictionInput.AveragePrice),
                    nameof(SalesPredictionInput.CustomerCount))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.Regression.Trainers.FastForest(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    numberOfTrees: 100,
                    numberOfLeaves: 20,
                    minimumExampleCountPerLeaf: 5));

            _logger.LogInformation("📊 تدريب النموذج على {Count} سجل...", trainingData.Count);
            _model = pipeline.Fit(splitData.TrainSet);

            // تقييم النموذج
            var predictions = _model.Transform(splitData.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            _logger.LogInformation("✅ تم التدريب بنجاح | R²={R2:F4} | MAE={MAE:F2} | RMSE={RMSE:F2}",
                metrics.RSquared, metrics.MeanAbsoluteError, metrics.RootMeanSquaredError);

            // حفظ النموذج
            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);
            _isModelLoaded = true;

            // إعادة بناء محرك التنبؤ
            _predictionEngine?.Dispose();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(_model);
        }

        // -----------------------------------------------
        // التنبؤ بالمبيعات للأشهر القادمة
        // -----------------------------------------------
        public async Task<List<SalesForecastResult>> ForecastNextMonthsAsync(int months = 6)
        {
            if (!_isModelLoaded)
                await TrainModelAsync();

            if (_predictionEngine == null)
                return GenerateFallbackForecast(months);

            var trainingData = await _trainingDataService.GetSalesTrainingDataAsync();
            var results = new List<SalesForecastResult>();
            var today = DateTime.Today;

            // حساب متوسطات تاريخية للمنتجات
            var avgByMonth = trainingData
                .GroupBy(d => d.Month)
                .ToDictionary(g => (int)g.Key, g => g.Average(d => d.ActualSales));

            var globalAvgSales = trainingData.Any() ? trainingData.Average(d => d.ActualSales) : 0f;
            var prevMonthSales = globalAvgSales;

            for (int i = 1; i <= months; i++)
            {
                var targetDate = today.AddMonths(i);
                var monthAvg = avgByMonth.TryGetValue(targetDate.Month, out float avg) ? avg : globalAvgSales;

                var input = new SalesPredictionInput
                {
                    Month = targetDate.Month,
                    Year = targetDate.Year,
                    DayOfWeek = 3f,
                    ProductId = 0,
                    CategoryId = 0,
                    PreviousMonthSales = prevMonthSales,
                    AveragePrice = trainingData.Any() ? trainingData.Average(d => d.AveragePrice) : 100f,
                    CustomerCount = trainingData.Any() ? trainingData.Average(d => d.CustomerCount) : 10f
                };

                var prediction = _predictionEngine.Predict(input);
                var predictedValue = Math.Max(0, prediction.PredictedSales);

                // حساب نطاق الثقة (±15%)
                var margin = predictedValue * 0.15f;

                results.Add(new SalesForecastResult
                {
                    ForecastDate = targetDate,
                    PredictedAmount = (decimal)predictedValue,
                    LowerBound = (decimal)Math.Max(0, predictedValue - margin),
                    UpperBound = (decimal)(predictedValue + margin),
                    ConfidenceLevel = 0.85,
                    Period = targetDate.ToString("MMMM yyyy")
                });

                prevMonthSales = predictedValue;
            }

            return results;
        }

        // -----------------------------------------------
        // التنبؤ لمنتج محدد
        // -----------------------------------------------
        public async Task<SalesForecastResult> ForecastForProductAsync(int productId, DateTime targetDate)
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var trainingData = await _trainingDataService.GetSalesTrainingDataAsync();
            var productData = trainingData.Where(d => (int)d.ProductId == productId).ToList();

            var input = new SalesPredictionInput
            {
                Month = targetDate.Month,
                Year = targetDate.Year,
                DayOfWeek = 3f,
                ProductId = productId,
                CategoryId = productData.Any() ? productData.First().CategoryId : 0,
                PreviousMonthSales = productData.Any() ? productData.Average(d => d.ActualSales) : 0f,
                AveragePrice = productData.Any() ? productData.Average(d => d.AveragePrice) : 100f,
                CustomerCount = productData.Any() ? productData.Average(d => d.CustomerCount) : 5f
            };

            float predictedValue = 0;
            if (_predictionEngine != null)
            {
                var prediction = _predictionEngine.Predict(input);
                predictedValue = Math.Max(0, prediction.PredictedSales);
            }

            var margin = predictedValue * 0.15f;
            return new SalesForecastResult
            {
                ForecastDate = targetDate,
                PredictedAmount = (decimal)predictedValue,
                LowerBound = (decimal)Math.Max(0, predictedValue - margin),
                UpperBound = (decimal)(predictedValue + margin),
                ConfidenceLevel = 0.80,
                Period = targetDate.ToString("MMMM yyyy")
            };
        }

        // -----------------------------------------------
        // التنبؤ حسب الفئة
        // -----------------------------------------------
        public async Task<List<SalesForecastResult>> ForecastByCategoryAsync(int categoryId, int months = 3)
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var results = new List<SalesForecastResult>();
            var today = DateTime.Today;

            for (int i = 1; i <= months; i++)
            {
                var targetDate = today.AddMonths(i);
                var result = await ForecastForProductAsync(0, targetDate);
                result.Period = $"فئة {categoryId} - {targetDate:MMMM yyyy}";
                results.Add(result);
            }

            return results;
        }

        // -----------------------------------------------
        // تقييم دقة النموذج
        // -----------------------------------------------
        public async Task<double> EvaluateModelAccuracyAsync()
        {
            if (!_isModelLoaded) return 0;

            var trainingData = await _trainingDataService.GetSalesTrainingDataAsync();
            if (trainingData.Count < 10) return 0;

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var predictions = _model!.Transform(splitData.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            return Math.Max(0, metrics.RSquared);
        }

        public Task<bool> IsModelTrainedAsync() => Task.FromResult(_isModelLoaded);

        // -----------------------------------------------
        // دوال مساعدة
        // -----------------------------------------------
        private void TryLoadSavedModel()
        {
            try
            {
                if (File.Exists(_modelPath))
                {
                    _model = _mlContext.Model.Load(_modelPath, out _);
                    _predictionEngine = _mlContext.Model.CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(_model);
                    _isModelLoaded = true;
                    _logger.LogInformation("✅ تم تحميل نموذج التنبؤ بالمبيعات من الملف.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ فشل تحميل النموذج المحفوظ: {Message}", ex.Message);
            }
        }

        private List<SalesForecastResult> GenerateFallbackForecast(int months)
        {
            var results = new List<SalesForecastResult>();
            var today = DateTime.Today;

            for (int i = 1; i <= months; i++)
            {
                var targetDate = today.AddMonths(i);
                results.Add(new SalesForecastResult
                {
                    ForecastDate = targetDate,
                    PredictedAmount = 0,
                    LowerBound = 0,
                    UpperBound = 0,
                    ConfidenceLevel = 0,
                    Period = targetDate.ToString("MMMM yyyy")
                });
            }

            return results;
        }
    }
}
