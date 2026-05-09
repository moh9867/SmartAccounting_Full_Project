using Microsoft.EntityFrameworkCore;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Models;

namespace SmartAccounting.AI.Data
{
    // ===================================================
    // خدمة استخراج بيانات التدريب من قاعدة البيانات
    // يجب ربط ISmartAccountingDbContext بـ DbContext الخاص بمشروعك
    // ===================================================
    public class TrainingDataService : ITrainingDataService
    {
        private readonly ISmartAccountingDbContext _context;

        public TrainingDataService(ISmartAccountingDbContext context)
        {
            _context = context;
        }

        // -----------------------------------------------
        // بيانات تدريب التنبؤ بالمبيعات
        // -----------------------------------------------
        public async Task<List<SalesPredictionInput>> GetSalesTrainingDataAsync()
        {
            // استخراج بيانات المبيعات الشهرية مع تجميعها
            var salesData = await _context.InvoiceItems
                .Include(ii => ii.Invoice)
                .Include(ii => ii.Product)
                .Where(ii => ii.Invoice.InvoiceDate >= DateTime.Now.AddYears(-3)
                          && ii.Invoice.InvoiceType == InvoiceType.Sales
                          && ii.Invoice.Status == InvoiceStatus.Posted)
                .GroupBy(ii => new
                {
                    Year = ii.Invoice.InvoiceDate.Year,
                    Month = ii.Invoice.InvoiceDate.Month,
                    ProductId = ii.ProductId,
                    CategoryId = ii.Product.CategoryId
                })
                .Select(g => new
                {
                    g.Key.Year,
                    g.Key.Month,
                    g.Key.ProductId,
                    g.Key.CategoryId,
                    TotalSales = g.Sum(ii => ii.Quantity * ii.UnitPrice),
                    AvgPrice = g.Average(ii => ii.UnitPrice),
                    CustomerCount = g.Select(ii => ii.Invoice.CustomerId).Distinct().Count()
                })
                .ToListAsync();

            var result = new List<SalesPredictionInput>();
            var previousSalesDict = new Dictionary<(int year, int month, int productId), float>();

            foreach (var sale in salesData.OrderBy(s => s.Year).ThenBy(s => s.Month))
            {
                var prevKey = (sale.Year, sale.Month - 1, sale.ProductId);
                if (sale.Month == 1) prevKey = (sale.Year - 1, 12, sale.ProductId);

                previousSalesDict.TryGetValue(prevKey, out float prevSales);

                var input = new SalesPredictionInput
                {
                    Month = sale.Month,
                    Year = sale.Year,
                    DayOfWeek = new DateTime(sale.Year, sale.Month, 1).DayOfWeek == DayOfWeek.Friday ? 1 : 0,
                    ProductId = sale.ProductId,
                    CategoryId = sale.CategoryId,
                    PreviousMonthSales = prevSales,
                    AveragePrice = (float)sale.AvgPrice,
                    CustomerCount = sale.CustomerCount,
                    ActualSales = (float)sale.TotalSales
                };

                result.Add(input);
                previousSalesDict[(sale.Year, sale.Month, sale.ProductId)] = (float)sale.TotalSales;
            }

            return result;
        }

        // -----------------------------------------------
        // بيانات تدريب تحليل مخاطر المخزون
        // -----------------------------------------------
        public async Task<List<InventoryRiskInput>> GetInventoryTrainingDataAsync()
        {
            var products = await _context.Products
                .Include(p => p.InvoiceItems)
                    .ThenInclude(ii => ii.Invoice)
                .Where(p => p.IsActive)
                .ToListAsync();

            var result = new List<InventoryRiskInput>();
            var today = DateTime.Today;

            foreach (var product in products)
            {
                var salesInvoices = product.InvoiceItems
                    .Where(ii => ii.Invoice.InvoiceType == InvoiceType.Sales
                              && ii.Invoice.Status == InvoiceStatus.Posted)
                    .ToList();

                if (!salesInvoices.Any()) continue;

                var lastSaleDate = salesInvoices.Max(ii => ii.Invoice.InvoiceDate);
                var daysSinceLastSale = (today - lastSaleDate).Days;

                var monthlySales = salesInvoices
                    .Where(ii => ii.Invoice.InvoiceDate >= today.AddMonths(-6))
                    .Sum(ii => ii.Quantity) / 6.0f;

                // تحديد المخزون الراكد: أكثر من 90 يوم بدون بيع أو مخزون أكبر من 12 شهر مبيعات
                var isSlowMoving = daysSinceLastSale > 90
                    || (monthlySales > 0 && product.CurrentStock / monthlySales > 12)
                    || (monthlySales == 0 && product.CurrentStock > 0);

                result.Add(new InventoryRiskInput
                {
                    ProductId = product.Id,
                    DaysSinceLastSale = daysSinceLastSale,
                    CurrentStock = (float)product.CurrentStock,
                    AverageMonthlySales = monthlySales,
                    PriceChangePercent = 0f, // يمكن ربطها بسجل الأسعار إن وجد
                    SeasonalityIndex = GetSeasonalityIndex(today.Month),
                    IsSlowMoving = isSlowMoving
                });
            }

            return result;
        }

        // -----------------------------------------------
        // بيانات تدريب التنبؤ بالتدفق النقدي
        // -----------------------------------------------
        public async Task<List<CashFlowPredictionInput>> GetCashFlowTrainingDataAsync()
        {
            var result = new List<CashFlowPredictionInput>();
            var startDate = DateTime.Now.AddYears(-2);

            for (var date = startDate; date < DateTime.Now; date = date.AddMonths(1))
            {
                var monthStart = new DateTime(date.Year, date.Month, 1);
                var monthEnd = monthStart.AddMonths(1).AddDays(-1);

                var receivables = await _context.CustomerAccounts
                    .Where(ca => ca.DueDate >= monthStart && ca.DueDate <= monthEnd && !ca.IsPaid)
                    .SumAsync(ca => (float?)ca.Amount) ?? 0f;

                var payables = await _context.SupplierAccounts
                    .Where(sa => sa.DueDate >= monthStart && sa.DueDate <= monthEnd && !sa.IsPaid)
                    .SumAsync(sa => (float?)sa.Amount) ?? 0f;

                var prevMonth = monthStart.AddMonths(-1);
                var prevBalance = await _context.JournalEntries
                    .Where(je => je.EntryDate < monthStart && je.AccountCode == "1010") // حساب النقدية
                    .SumAsync(je => (float?)(je.DebitAmount - je.CreditAmount)) ?? 0f;

                var netCashFlow = await _context.JournalEntries
                    .Where(je => je.EntryDate >= monthStart && je.EntryDate <= monthEnd && je.AccountCode == "1010")
                    .SumAsync(je => (float?)(je.DebitAmount - je.CreditAmount)) ?? 0f;

                result.Add(new CashFlowPredictionInput
                {
                    Month = date.Month,
                    Year = date.Year,
                    TotalReceivables = receivables,
                    TotalPayables = payables,
                    PreviousCashBalance = prevBalance,
                    AverageCollectionDays = 30f,
                    AveragePaymentDays = 45f,
                    NetCashFlow = netCashFlow
                });
            }

            return result;
        }

        // -----------------------------------------------
        // بيانات تدريب تقسيم العملاء
        // -----------------------------------------------
        public async Task<List<CustomerSegmentInput>> GetCustomerTrainingDataAsync()
        {
            var customers = await _context.Customers
                .Include(c => c.Invoices)
                    .ThenInclude(i => i.InvoiceItems)
                .Where(c => c.IsActive && c.Invoices.Any())
                .ToListAsync();

            var result = new List<CustomerSegmentInput>();
            var today = DateTime.Today;

            foreach (var customer in customers)
            {
                var postedInvoices = customer.Invoices
                    .Where(i => i.InvoiceType == InvoiceType.Sales && i.Status == InvoiceStatus.Posted)
                    .ToList();

                if (!postedInvoices.Any()) continue;

                var totalPurchases = postedInvoices.Sum(i => i.TotalAmount);
                var frequency = postedInvoices.Count;
                var lastPurchase = postedInvoices.Max(i => i.InvoiceDate);
                var daysSinceLast = (today - lastPurchase).Days;
                var avgOrderValue = totalPurchases / frequency;

                // تحديد الشريحة بناء على RFM (Recency, Frequency, Monetary)
                uint segment;
                if (totalPurchases > 50000 && frequency > 20 && daysSinceLast < 30)
                    segment = (uint)CustomerSegment.VIP;
                else if (daysSinceLast > 180)
                    segment = (uint)CustomerSegment.Lost;
                else if (daysSinceLast > 90 || frequency < 3)
                    segment = (uint)CustomerSegment.AtRisk;
                else
                    segment = (uint)CustomerSegment.Regular;

                result.Add(new CustomerSegmentInput
                {
                    TotalPurchases = (float)totalPurchases,
                    PurchaseFrequency = frequency,
                    DaysSinceLastPurchase = daysSinceLast,
                    AverageOrderValue = (float)avgOrderValue,
                    PaymentOnTimeRate = 0.8f,
                    Segment = segment
                });
            }

            return result;
        }

        // -----------------------------------------------
        // بيانات كشف الشذوذ في المعاملات
        // -----------------------------------------------
        public async Task<List<AnomalyDetectionInput>> GetTransactionDataForAnomalyAsync()
        {
            var amounts = await _context.JournalEntries
                .Where(je => je.EntryDate >= DateTime.Now.AddYears(-1))
                .OrderBy(je => je.EntryDate)
                .Select(je => new AnomalyDetectionInput
                {
                    TransactionAmount = (float)je.DebitAmount
                })
                .ToListAsync();

            return amounts;
        }

        // -----------------------------------------------
        // دالة مساعدة: مؤشر الموسمية
        // -----------------------------------------------
        private float GetSeasonalityIndex(int month) => month switch
        {
            11 or 12 => 1.3f,   // موسم نهاية العام
            1 or 2   => 0.8f,   // ركود بداية السنة
            6 or 7   => 0.9f,   // الصيف
            9 or 10  => 1.1f,   // بداية الموسم
            _        => 1.0f
        };
    }

    // ===================================================
    // واجهة DbContext - ربطها بـ DbContext الخاص بمشروعك
    // ===================================================
    public interface ISmartAccountingDbContext
    {
        DbSet<Product> Products { get; }
        DbSet<Invoice> Invoices { get; }
        DbSet<InvoiceItem> InvoiceItems { get; }
        DbSet<Customer> Customers { get; }
        DbSet<JournalEntry> JournalEntries { get; }
        DbSet<CustomerAccount> CustomerAccounts { get; }
        DbSet<SupplierAccount> SupplierAccounts { get; }
        Task<int> SaveChangesAsync(CancellationToken cancellationToken = default);
    }

    // ===================================================
    // نماذج قاعدة البيانات (Entity Stubs)
    // استبدلها بنماذجك الفعلية في المشروع
    // ===================================================
    public class Product
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public string Code { get; set; } = string.Empty;
        public int CategoryId { get; set; }
        public decimal CurrentStock { get; set; }
        public decimal CostPrice { get; set; }
        public decimal SalePrice { get; set; }
        public bool IsActive { get; set; }
        public virtual ICollection<InvoiceItem> InvoiceItems { get; set; } = new List<InvoiceItem>();
    }

    public class Invoice
    {
        public int Id { get; set; }
        public string InvoiceNumber { get; set; } = string.Empty;
        public DateTime InvoiceDate { get; set; }
        public int? CustomerId { get; set; }
        public InvoiceType InvoiceType { get; set; }
        public InvoiceStatus Status { get; set; }
        public decimal TotalAmount { get; set; }
        public virtual Customer? Customer { get; set; }
        public virtual ICollection<InvoiceItem> InvoiceItems { get; set; } = new List<InvoiceItem>();
    }

    public class InvoiceItem
    {
        public int Id { get; set; }
        public int InvoiceId { get; set; }
        public int ProductId { get; set; }
        public decimal Quantity { get; set; }
        public decimal UnitPrice { get; set; }
        public virtual Invoice Invoice { get; set; } = null!;
        public virtual Product Product { get; set; } = null!;
    }

    public class Customer
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public bool IsActive { get; set; }
        public virtual ICollection<Invoice> Invoices { get; set; } = new List<Invoice>();
    }

    public class JournalEntry
    {
        public int Id { get; set; }
        public DateTime EntryDate { get; set; }
        public string AccountCode { get; set; } = string.Empty;
        public decimal DebitAmount { get; set; }
        public decimal CreditAmount { get; set; }
        public string Description { get; set; } = string.Empty;
    }

    public class CustomerAccount
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public decimal Amount { get; set; }
        public DateTime DueDate { get; set; }
        public bool IsPaid { get; set; }
    }

    public class SupplierAccount
    {
        public int Id { get; set; }
        public int SupplierId { get; set; }
        public decimal Amount { get; set; }
        public DateTime DueDate { get; set; }
        public bool IsPaid { get; set; }
    }

    public enum InvoiceType { Sales, Purchase, SalesReturn, PurchaseReturn }
    public enum InvoiceStatus { Draft, Posted, Cancelled }
}
