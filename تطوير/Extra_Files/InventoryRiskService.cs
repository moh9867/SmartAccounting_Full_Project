using Microsoft.Extensions.Logging;
using Microsoft.ML;
using SmartAccounting.AI.Data;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Services
{
    // ===================================================
    // خدمة تحليل مخاطر المخزون الراكد باستخدام ML.NET
    // ===================================================
    public class InventoryRiskService : IInventoryRiskService
    {
        private readonly MLContext _mlContext;
        private readonly ITrainingDataService _trainingDataService;
        private readonly ISmartAccountingDbContext _context;
        private readonly ILogger<InventoryRiskService> _logger;
        private readonly string _modelPath;

        private ITransformer? _model;
        private PredictionEngine<InventoryRiskInput, InventoryRiskOutput>? _predictionEngine;
        private bool _isModelLoaded = false;

        public InventoryRiskService(
            ITrainingDataService trainingDataService,
            ISmartAccountingDbContext context,
            ILogger<InventoryRiskService> logger,
            string modelStoragePath = "MLModels")
        {
            _mlContext = new MLContext(seed: 42);
            _trainingDataService = trainingDataService;
            _context = context;
            _logger = logger;
            _modelPath = Path.Combine(modelStoragePath, "inventory_risk_model.zip");

            Directory.CreateDirectory(modelStoragePath);
            TryLoadSavedModel();
        }

        // -----------------------------------------------
        // تدريب نموذج تحليل مخاطر المخزون
        // -----------------------------------------------
        public async Task TrainModelAsync()
        {
            _logger.LogInformation("🤖 بدء تدريب نموذج تحليل مخاطر المخزون...");

            var trainingData = await _trainingDataService.GetInventoryTrainingDataAsync();
            if (trainingData.Count < 20)
            {
                _logger.LogWarning("⚠️ بيانات غير كافية ({Count} منتج).", trainingData.Count);
                return;
            }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = _mlContext.Transforms
                .Concatenate("Features",
                    nameof(InventoryRiskInput.DaysSinceLastSale),
                    nameof(InventoryRiskInput.CurrentStock),
                    nameof(InventoryRiskInput.AverageMonthlySales),
                    nameof(InventoryRiskInput.PriceChangePercent),
                    nameof(InventoryRiskInput.SeasonalityIndex))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    numberOfTrees: 50,
                    numberOfLeaves: 20,
                    minimumExampleCountPerLeaf: 3));

            _model = pipeline.Fit(splitData.TrainSet);

            var predictions = _model.Transform(splitData.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

            _logger.LogInformation("✅ تدريب ناجح | دقة={Accuracy:P1} | AUC={AUC:F4}",
                metrics.Accuracy, metrics.AreaUnderRocCurve);

            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);
            _isModelLoaded = true;

            _predictionEngine?.Dispose();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<InventoryRiskInput, InventoryRiskOutput>(_model);
        }

        // -----------------------------------------------
        // تحليل مخاطر جميع المنتجات
        // -----------------------------------------------
        public async Task<List<InventoryRiskResult>> GetAllInventoryRisksAsync()
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var products = _context.Products
                .Where(p => p.IsActive && p.CurrentStock > 0)
                .ToList();

            var results = new List<InventoryRiskResult>();

            foreach (var product in products)
            {
                var risk = await AnalyzeProductRiskAsync(product.Id);
                results.Add(risk);
            }

            return results.OrderByDescending(r => r.RiskScore).ToList();
        }

        // -----------------------------------------------
        // الحصول على العناصر ذات المخاطر العالية
        // -----------------------------------------------
        public async Task<List<InventoryRiskResult>> GetHighRiskItemsAsync(float minRiskScore = 0.7f)
        {
            var allRisks = await GetAllInventoryRisksAsync();
            return allRisks.Where(r => r.RiskScore >= minRiskScore).ToList();
        }

        // -----------------------------------------------
        // تحليل مخاطر منتج محدد
        // -----------------------------------------------
        public async Task<InventoryRiskResult> AnalyzeProductRiskAsync(int productId)
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var product = _context.Products.FirstOrDefault(p => p.Id == productId);
            if (product == null)
                return new InventoryRiskResult { ProductId = productId, RiskLevel = RiskLevel.Low };

            var trainingData = await _trainingDataService.GetInventoryTrainingDataAsync();
            var productData = trainingData.FirstOrDefault(d => (int)d.ProductId == productId);

            if (productData == null)
            {
                return new InventoryRiskResult
                {
                    ProductId = productId,
                    ProductName = product.Name,
                    ProductCode = product.Code,
                    CurrentStock = product.CurrentStock,
                    CurrentValue = product.CurrentStock * product.CostPrice,
                    RiskScore = 0.5f,
                    RiskLevel = RiskLevel.Medium,
                    Recommendation = "لا توجد بيانات مبيعات كافية للتحليل"
                };
            }

            float riskScore = 0.5f;
            bool isSlowMoving = false;

            if (_predictionEngine != null)
            {
                var prediction = _predictionEngine.Predict(productData);
                riskScore = prediction.RiskProbability;
                isSlowMoving = prediction.IsSlowMoving;
            }
            else
            {
                // تحليل بسيط بدون النموذج
                riskScore = CalculateSimpleRiskScore(productData);
                isSlowMoving = riskScore > 0.5f;
            }

            var riskLevel = riskScore switch
            {
                >= 0.85f => RiskLevel.Critical,
                >= 0.70f => RiskLevel.High,
                >= 0.40f => RiskLevel.Medium,
                _        => RiskLevel.Low
            };

            var monthsToDeplete = productData.AverageMonthlySales > 0
                ? (int)Math.Ceiling(productData.CurrentStock / productData.AverageMonthlySales)
                : 999;

            return new InventoryRiskResult
            {
                ProductId = productId,
                ProductName = product.Name,
                ProductCode = product.Code,
                CurrentStock = product.CurrentStock,
                CurrentValue = product.CurrentStock * product.CostPrice,
                RiskScore = riskScore,
                RiskLevel = riskLevel,
                DaysSinceLastSale = (int)productData.DaysSinceLastSale,
                AverageMonthlySales = (decimal)productData.AverageMonthlySales,
                EstimatedMonthsToDeplete = monthsToDeplete,
                Recommendation = BuildRecommendation(riskLevel, monthsToDeplete, productData)
            };
        }

        // -----------------------------------------------
        // توزيع المخاطر حسب المستوى
        // -----------------------------------------------
        public async Task<Dictionary<RiskLevel, int>> GetRiskDistributionAsync()
        {
            var allRisks = await GetAllInventoryRisksAsync();
            return allRisks
                .GroupBy(r => r.RiskLevel)
                .ToDictionary(g => g.Key, g => g.Count());
        }

        // -----------------------------------------------
        // إجمالي قيمة المخزون المعرض للخطر
        // -----------------------------------------------
        public async Task<decimal> GetTotalAtRiskValueAsync()
        {
            var highRisk = await GetHighRiskItemsAsync(0.7f);
            return highRisk.Sum(r => r.CurrentValue);
        }

        // -----------------------------------------------
        // دوال مساعدة
        // -----------------------------------------------
        private float CalculateSimpleRiskScore(InventoryRiskInput data)
        {
            float score = 0f;

            if (data.DaysSinceLastSale > 180) score += 0.4f;
            else if (data.DaysSinceLastSale > 90) score += 0.25f;
            else if (data.DaysSinceLastSale > 30) score += 0.1f;

            if (data.AverageMonthlySales > 0)
            {
                var monthsCover = data.CurrentStock / data.AverageMonthlySales;
                if (monthsCover > 12) score += 0.4f;
                else if (monthsCover > 6) score += 0.2f;
            }
            else if (data.CurrentStock > 0)
            {
                score += 0.3f;
            }

            return Math.Min(1f, score);
        }

        private string BuildRecommendation(RiskLevel level, int monthsToDeplete, InventoryRiskInput data) =>
            level switch
            {
                RiskLevel.Critical => $"⚠️ تصفية عاجلة! المخزون راكد منذ {data.DaysSinceLastSale} يوم. طبّق خصماً بنسبة 30-50% لتسريع البيع.",
                RiskLevel.High     => $"🔴 مخاطر عالية. يتبقى ~{monthsToDeplete} شهر لنفاد المخزون. ابدأ حملة ترويجية فورية.",
                RiskLevel.Medium   => $"🟡 مراقبة دورية. آخر بيع قبل {data.DaysSinceLastSale} يوم. راجع استراتيجية التسعير.",
                _                  => $"✅ المخزون في وضع جيد. متوسط المبيعات الشهري: {data.AverageMonthlySales:F0} وحدة."
            };

        private void TryLoadSavedModel()
        {
            try
            {
                if (File.Exists(_modelPath))
                {
                    _model = _mlContext.Model.Load(_modelPath, out _);
                    _predictionEngine = _mlContext.Model.CreatePredictionEngine<InventoryRiskInput, InventoryRiskOutput>(_model);
                    _isModelLoaded = true;
                    _logger.LogInformation("✅ تم تحميل نموذج المخزون من الملف.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ فشل تحميل نموذج المخزون: {Message}", ex.Message);
            }
        }
    }
}
