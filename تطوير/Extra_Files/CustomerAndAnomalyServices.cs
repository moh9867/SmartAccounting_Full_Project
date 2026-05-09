using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using SmartAccounting.AI.Data;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Services
{
    // ===================================================
    // خدمة تقسيم العملاء (RFM + ML)
    // ===================================================
    public class CustomerSegmentService : ICustomerSegmentService
    {
        private readonly MLContext _mlContext;
        private readonly ITrainingDataService _trainingDataService;
        private readonly ISmartAccountingDbContext _context;
        private readonly ILogger<CustomerSegmentService> _logger;
        private readonly string _modelPath;

        private ITransformer? _model;
        private PredictionEngine<CustomerSegmentInput, CustomerSegmentOutput>? _predictionEngine;
        private bool _isModelLoaded = false;

        public CustomerSegmentService(
            ITrainingDataService trainingDataService,
            ISmartAccountingDbContext context,
            ILogger<CustomerSegmentService> logger,
            string modelStoragePath = "MLModels")
        {
            _mlContext = new MLContext(seed: 42);
            _trainingDataService = trainingDataService;
            _context = context;
            _logger = logger;
            _modelPath = Path.Combine(modelStoragePath, "customer_segment_model.zip");
            Directory.CreateDirectory(modelStoragePath);
            TryLoadSavedModel();
        }

        public async Task TrainModelAsync()
        {
            _logger.LogInformation("🤖 تدريب نموذج تقسيم العملاء...");

            var trainingData = await _trainingDataService.GetCustomerTrainingDataAsync();
            if (trainingData.Count < 10) { _logger.LogWarning("⚠️ بيانات عملاء غير كافية."); return; }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = _mlContext.Transforms
                .Concatenate("Features",
                    nameof(CustomerSegmentInput.TotalPurchases),
                    nameof(CustomerSegmentInput.PurchaseFrequency),
                    nameof(CustomerSegmentInput.DaysSinceLastPurchase),
                    nameof(CustomerSegmentInput.AverageOrderValue),
                    nameof(CustomerSegmentInput.PaymentOnTimeRate))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Label",
                    featureColumnName: "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _model = pipeline.Fit(splitData.TrainSet);

            var predictions = _model.Transform(splitData.TestSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");
            _logger.LogInformation("✅ تقسيم العملاء | دقة={Accuracy:P1}", metrics.MacroAccuracy);

            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);
            _isModelLoaded = true;

            _predictionEngine?.Dispose();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<CustomerSegmentInput, CustomerSegmentOutput>(_model);
        }

        public async Task<List<CustomerSegmentResult>> SegmentAllCustomersAsync()
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var customers = _context.Customers
                .Include(c => c.Invoices)
                .Where(c => c.IsActive)
                .ToList();

            var results = new List<CustomerSegmentResult>();
            foreach (var customer in customers)
            {
                var result = await SegmentCustomerAsync(customer.Id);
                results.Add(result);
            }

            return results.OrderBy(r => r.Segment).ToList();
        }

        public async Task<CustomerSegmentResult> SegmentCustomerAsync(int customerId)
        {
            if (!_isModelLoaded) await TrainModelAsync();

            var customer = _context.Customers
                .Include(c => c.Invoices)
                .FirstOrDefault(c => c.Id == customerId);

            if (customer == null) return new CustomerSegmentResult { CustomerId = customerId };

            var postedInvoices = customer.Invoices
                .Where(i => i.InvoiceType == InvoiceType.Sales && i.Status == InvoiceStatus.Posted)
                .ToList();

            if (!postedInvoices.Any())
                return new CustomerSegmentResult
                {
                    CustomerId = customerId,
                    CustomerName = customer.Name,
                    Segment = CustomerSegment.Lost,
                    SegmentLabel = "عميل بدون مشتريات",
                    RecommendedAction = "📞 تواصل مع العميل وأرسل له عرضاً خاصاً."
                };

            var totalPurchases = postedInvoices.Sum(i => i.TotalAmount);
            var frequency = postedInvoices.Count;
            var lastPurchase = postedInvoices.Max(i => i.InvoiceDate);
            var daysSinceLast = (DateTime.Today - lastPurchase).Days;
            var avgOrderValue = totalPurchases / frequency;

            var input = new CustomerSegmentInput
            {
                TotalPurchases = (float)totalPurchases,
                PurchaseFrequency = frequency,
                DaysSinceLastPurchase = daysSinceLast,
                AverageOrderValue = (float)avgOrderValue,
                PaymentOnTimeRate = 0.8f
            };

            CustomerSegment segment;
            if (_predictionEngine != null)
            {
                var prediction = _predictionEngine.Predict(input);
                segment = (CustomerSegment)prediction.PredictedSegment;
            }
            else
            {
                segment = ClassifyManually(totalPurchases, frequency, daysSinceLast);
            }

            return new CustomerSegmentResult
            {
                CustomerId = customerId,
                CustomerName = customer.Name,
                Segment = segment,
                SegmentLabel = GetSegmentLabel(segment),
                TotalPurchases = totalPurchases,
                PurchaseFrequency = frequency,
                DaysSinceLastPurchase = daysSinceLast,
                RecommendedAction = GetRecommendedAction(segment, daysSinceLast, totalPurchases)
            };
        }

        public async Task<Dictionary<CustomerSegment, List<CustomerSegmentResult>>> GetSegmentedCustomersAsync()
        {
            var all = await SegmentAllCustomersAsync();
            return all.GroupBy(c => c.Segment)
                      .ToDictionary(g => g.Key, g => g.ToList());
        }

        public async Task<List<CustomerSegmentResult>> GetAtRiskCustomersAsync()
        {
            var all = await SegmentAllCustomersAsync();
            return all.Where(c => c.Segment == CustomerSegment.AtRisk || c.Segment == CustomerSegment.Lost).ToList();
        }

        private CustomerSegment ClassifyManually(decimal total, int freq, int days)
        {
            if (total > 50000 && freq > 20 && days < 30) return CustomerSegment.VIP;
            if (days > 180) return CustomerSegment.Lost;
            if (days > 90 || freq < 3) return CustomerSegment.AtRisk;
            return CustomerSegment.Regular;
        }

        private string GetSegmentLabel(CustomerSegment segment) => segment switch
        {
            CustomerSegment.VIP     => "⭐ عميل VIP",
            CustomerSegment.Regular => "✅ عميل منتظم",
            CustomerSegment.AtRisk  => "⚠️ عميل في خطر",
            CustomerSegment.Lost    => "❌ عميل مفقود",
            _                       => "غير محدد"
        };

        private string GetRecommendedAction(CustomerSegment segment, int days, decimal total) => segment switch
        {
            CustomerSegment.VIP     => "🎁 أرسل برنامج ولاء حصري وخصم 10% على الطلب القادم.",
            CustomerSegment.Regular => "📧 أرسل نشرة إخبارية بالعروض الشهرية للحفاظ على الانتظام.",
            CustomerSegment.AtRisk  => $"🔔 لم يشترِ منذ {days} يوماً. أرسل عرضاً خاصاً بخصم 15% الآن.",
            CustomerSegment.Lost    => $"📞 عميل غير نشط منذ {days} يوماً. جرّب حملة إعادة اكتساب أو استطلاع رأي.",
            _                       => "راجع بيانات العميل."
        };

        private void TryLoadSavedModel()
        {
            try
            {
                if (File.Exists(_modelPath))
                {
                    _model = _mlContext.Model.Load(_modelPath, out _);
                    _predictionEngine = _mlContext.Model.CreatePredictionEngine<CustomerSegmentInput, CustomerSegmentOutput>(_model);
                    _isModelLoaded = true;
                }
            }
            catch { /* سيتم التدريب عند الحاجة */ }
        }
    }

    // ===================================================
    // خدمة كشف الشذوذ في المعاملات المالية
    // ===================================================
    public class AnomalyDetectionService : IAnomalyDetectionService
    {
        private readonly MLContext _mlContext;
        private readonly ITrainingDataService _trainingDataService;
        private readonly ISmartAccountingDbContext _context;
        private readonly ILogger<AnomalyDetectionService> _logger;

        private ITransformer? _model;
        private bool _isModelLoaded = false;

        public AnomalyDetectionService(
            ITrainingDataService trainingDataService,
            ISmartAccountingDbContext context,
            ILogger<AnomalyDetectionService> logger)
        {
            _mlContext = new MLContext(seed: 42);
            _trainingDataService = trainingDataService;
            _context = context;
            _logger = logger;
        }

        public async Task TrainModelAsync()
        {
            _logger.LogInformation("🤖 تدريب نموذج كشف الشذوذ...");

            var trainingData = await _trainingDataService.GetTransactionDataForAnomalyAsync();
            if (trainingData.Count < 30) { _logger.LogWarning("⚠️ بيانات شذوذ غير كافية."); return; }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);

            var pipeline = _mlContext.Transforms.DetectIidSpike(
                outputColumnName: "Prediction",
                inputColumnName: nameof(AnomalyDetectionInput.TransactionAmount),
                confidence: 95,
                pvalueHistoryLength: trainingData.Count / 4);

            _model = pipeline.Fit(dataView);
            _isModelLoaded = true;
            _logger.LogInformation("✅ نموذج كشف الشذوذ جاهز.");
        }

        public async Task<List<TransactionAnomalyResult>> DetectAnomaliesAsync(DateTime? fromDate = null)
        {
            if (!_isModelLoaded) await TrainModelAsync();
            if (_model == null) return new List<TransactionAnomalyResult>();

            var from = fromDate ?? DateTime.Now.AddMonths(-3);

            var transactions = _context.JournalEntries
                .Where(je => je.EntryDate >= from)
                .OrderBy(je => je.EntryDate)
                .ToList();

            if (!transactions.Any()) return new List<TransactionAnomalyResult>();

            var inputData = transactions.Select(t => new AnomalyDetectionInput
            {
                TransactionAmount = (float)t.DebitAmount
            }).ToList();

            var dataView = _mlContext.Data.LoadFromEnumerable(inputData);
            var predictions = _model.Transform(dataView);
            var anomalyPredictions = _mlContext.Data
                .CreateEnumerable<AnomalyDetectionOutput>(predictions, reuseRowObject: false)
                .ToList();

            var results = new List<TransactionAnomalyResult>();
            var avgAmount = transactions.Average(t => (double)t.DebitAmount);

            for (int i = 0; i < Math.Min(transactions.Count, anomalyPredictions.Count); i++)
            {
                if (!anomalyPredictions[i].IsAnomaly) continue;

                var tx = transactions[i];
                var deviation = Math.Abs((double)tx.DebitAmount - avgAmount) / avgAmount * 100;

                results.Add(new TransactionAnomalyResult
                {
                    TransactionId = tx.Id,
                    TransactionDate = tx.EntryDate,
                    Description = tx.Description,
                    Amount = tx.DebitAmount,
                    IsAnomaly = true,
                    AnomalyScore = anomalyPredictions[i].AnomalyScore,
                    ExpectedAmount = avgAmount,
                    AnomalyReason = BuildAnomalyReason(tx.DebitAmount, (decimal)avgAmount, deviation)
                });
            }

            return results.OrderByDescending(r => r.AnomalyScore).ToList();
        }

        public async Task<TransactionAnomalyResult> AnalyzeTransactionAsync(int transactionId)
        {
            var tx = _context.JournalEntries.FirstOrDefault(je => je.Id == transactionId);
            if (tx == null) return new TransactionAnomalyResult { TransactionId = transactionId };

            var allAnomalies = await DetectAnomaliesAsync();
            return allAnomalies.FirstOrDefault(a => a.TransactionId == transactionId)
                ?? new TransactionAnomalyResult
                {
                    TransactionId = transactionId,
                    TransactionDate = tx.EntryDate,
                    Description = tx.Description,
                    Amount = tx.DebitAmount,
                    IsAnomaly = false
                };
        }

        public async Task<List<TransactionAnomalyResult>> GetRecentAnomaliesAsync(int count = 10)
        {
            var all = await DetectAnomaliesAsync(DateTime.Now.AddMonths(-1));
            return all.Take(count).ToList();
        }

        private string BuildAnomalyReason(decimal amount, decimal avg, double deviation) =>
            amount > avg
                ? $"📈 مبلغ مرتفع بشكل غير طبيعي ({deviation:F1}% فوق المتوسط {avg:N0}). تحقق من صحة الإدخال."
                : $"📉 مبلغ منخفض بشكل غير طبيعي ({deviation:F1}% تحت المتوسط {avg:N0}). قد يكون خطأ إدخال.";
    }
}
