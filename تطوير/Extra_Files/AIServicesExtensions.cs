using Microsoft.Extensions.DependencyInjection;
using SmartAccounting.AI.Data;
using SmartAccounting.AI.Interfaces;
using SmartAccounting.AI.Services;

namespace SmartAccounting.AI.Extensions
{
    // ===================================================
    // تسجيل جميع خدمات الذكاء الاصطناعي في DI Container
    // ===================================================
    public static class AIServicesExtensions
    {
        /// <summary>
        /// أضف هذا في Program.cs أو Startup.cs:
        /// builder.Services.AddSmartAccountingAI();
        /// </summary>
        public static IServiceCollection AddSmartAccountingAI(
            this IServiceCollection services,
            string modelStoragePath = "MLModels")
        {
            // خدمة استخراج بيانات التدريب
            services.AddScoped<ITrainingDataService, TrainingDataService>();

            // خدمات التنبؤ الأساسية
            services.AddSingleton<ISalesForecastService>(sp =>
                new SalesForecastService(
                    sp.GetRequiredService<ITrainingDataService>(),
                    sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<SalesForecastService>>(),
                    modelStoragePath));

            services.AddSingleton<IInventoryRiskService>(sp =>
                new InventoryRiskService(
                    sp.GetRequiredService<ITrainingDataService>(),
                    sp.GetRequiredService<ISmartAccountingDbContext>(),
                    sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<InventoryRiskService>>(),
                    modelStoragePath));

            services.AddSingleton<ICashFlowForecastService>(sp =>
                new CashFlowForecastService(
                    sp.GetRequiredService<ITrainingDataService>(),
                    sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<CashFlowForecastService>>(),
                    modelStoragePath));

            services.AddSingleton<ICustomerSegmentService>(sp =>
                new CustomerSegmentService(
                    sp.GetRequiredService<ITrainingDataService>(),
                    sp.GetRequiredService<ISmartAccountingDbContext>(),
                    sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<CustomerSegmentService>>(),
                    modelStoragePath));

            services.AddSingleton<IAnomalyDetectionService>(sp =>
                new AnomalyDetectionService(
                    sp.GetRequiredService<ITrainingDataService>(),
                    sp.GetRequiredService<ISmartAccountingDbContext>(),
                    sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<AnomalyDetectionService>>()));

            // لوحة التحكم الرئيسية (تجمع كل الخدمات)
            services.AddSingleton<IAIDashboardService>(sp =>
                new AIDashboardService(
                    sp.GetRequiredService<ISalesForecastService>(),
                    sp.GetRequiredService<IInventoryRiskService>(),
                    sp.GetRequiredService<ICashFlowForecastService>(),
                    sp.GetRequiredService<ICustomerSegmentService>(),
                    sp.GetRequiredService<IAnomalyDetectionService>(),
                    sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<AIDashboardService>>(),
                    modelStoragePath));

            return services;
        }
    }
}
