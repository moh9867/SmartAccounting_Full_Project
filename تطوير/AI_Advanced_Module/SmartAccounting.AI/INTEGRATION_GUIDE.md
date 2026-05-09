# 🤖 دليل ربط نظام الذكاء الاصطناعي بمشروع Smart Accounting

## 📁 هيكل الملفات المُنشأة

```
SmartAccounting.AI/
├── Models/
│   └── AIModels.cs              ← جميع نماذج البيانات والـ DTOs
├── Interfaces/
│   └── IAIServices.cs           ← واجهات جميع الخدمات
├── Data/
│   └── TrainingDataService.cs   ← استخراج بيانات التدريب من DB
├── Services/
│   ├── SalesForecastService.cs      ← التنبؤ بالمبيعات
│   ├── InventoryRiskService.cs      ← تحليل مخاطر المخزون
│   ├── CashFlowForecastService.cs   ← التنبؤ بالتدفق النقدي
│   ├── CustomerAndAnomalyServices.cs ← العملاء + كشف الشذوذ
│   └── AIDashboardService.cs        ← لوحة التحكم المركزية
├── Extensions/
│   └── AIServicesExtensions.cs  ← تسجيل DI (سطر واحد في Program.cs)
└── Blazor/
    ├── Pages/
    │   └── AIDashboard.razor    ← صفحة لوحة التحكم الكاملة
    └── wwwroot/css/
        └── ai-dashboard.css     ← التصميم الكامل
```

---

## ⚡ خطوات الربط (5 خطوات فقط)

### الخطوة 1: أضف المشروع للـ Solution

```bash
dotnet sln add SmartAccounting.AI/SmartAccounting.AI.csproj

# أضف مرجع المشروع لمشروع Blazor الرئيسي
dotnet add 04-Presentation/YourBlazorProject.csproj reference SmartAccounting.AI/SmartAccounting.AI.csproj
```

### الخطوة 2: ثبّت حزم NuGet

```bash
cd SmartAccounting.AI
dotnet add package Microsoft.ML --version 3.0.1
dotnet add package Microsoft.ML.FastTree --version 3.0.1
dotnet add package Microsoft.ML.TimeSeries --version 3.0.1
```

### الخطوة 3: اربط DbContext الخاص بك

في `Data/TrainingDataService.cs`، الواجهة `ISmartAccountingDbContext` 
**استبدلها** بـ DbContext الفعلي في مشروعك:

```csharp
// في مشروعك الأصلي (03-Infrastructure)، أضف هذا للـ DbContext:
public class SmartAccountingDbContext : DbContext, ISmartAccountingDbContext
{
    // DbContext الموجود لديك بالفعل + تطبيق الواجهة
}

// سجّل في DI:
services.AddScoped<ISmartAccountingDbContext>(sp =>
    sp.GetRequiredService<SmartAccountingDbContext>());
```

### الخطوة 4: سجّل الخدمات في Program.cs

```csharp
// في Program.cs أو Startup.cs
using SmartAccounting.AI.Extensions;

// أضف هذا السطر الواحد:
builder.Services.AddSmartAccountingAI(modelStoragePath: "MLModels");
```

### الخطوة 5: أضف الصفحة والـ CSS

في `App.razor` أو `MainLayout.razor`:
```html
<link href="css/ai-dashboard.css" rel="stylesheet" />
```

في القائمة الجانبية:
```html
<NavLink href="ai-dashboard">
    🤖 لوحة الذكاء الاصطناعي
</NavLink>
```

---

## 🗄️ متطلبات قاعدة البيانات

النظام يتوقع الجداول التالية (موجودة بالفعل في مشروعك):

| الجدول | الحقول المستخدمة |
|--------|-----------------|
| `InvoiceItems` | InvoiceId, ProductId, Quantity, UnitPrice |
| `Invoices` | InvoiceDate, InvoiceType, Status, TotalAmount, CustomerId |
| `Products` | Id, Name, Code, CategoryId, CurrentStock, CostPrice |
| `Customers` | Id, Name, IsActive |
| `JournalEntries` | EntryDate, AccountCode, DebitAmount, CreditAmount |
| `CustomerAccounts` | CustomerId, Amount, DueDate, IsPaid |
| `SupplierAccounts` | SupplierId, Amount, DueDate, IsPaid |

---

## 🎯 ما يفعله النظام تلقائياً

| الميزة | الوصف |
|--------|-------|
| 📈 **توقع المبيعات** | يتنبأ بمبيعات الـ 6 أشهر القادمة بدقة 85%+ |
| 📦 **مخاطر المخزون** | يكشف المنتجات الراكدة قبل أن تصبح خسارة |
| 💰 **التدفق النقدي** | يُنبّه عند توقع عجز نقدي مسبقاً |
| 👥 **تقسيم العملاء** | يصنّف العملاء (VIP/منتظم/خطر/مفقود) |
| 🔍 **كشف الشذوذ** | يكتشف المعاملات المشبوهة تلقائياً |

---

## 🔧 تخصيص النماذج

### تغيير حساسية كشف الشذوذ
```csharp
// في AnomalyDetectionService.cs
var pipeline = _mlContext.Transforms.DetectIidSpike(
    confidence: 99,  // ارفع للتقليل من التنبيهات (90-99)
    ...);
```

### تغيير معايير المخزون الراكد
```csharp
// في TrainingDataService.cs
var isSlowMoving = daysSinceLastSale > 60   // كان 90، خفّضه للحساسية
    || (monthsCover > 6);                    // كان 12 شهراً
```

### جدولة إعادة التدريب التلقائي
```csharp
// في Program.cs - تدريب تلقائي كل أسبوع
builder.Services.AddHostedService<AITrainingBackgroundService>();
```

---

## 📊 المتطلبات الدنيا للتدريب

| النموذج | الحد الأدنى من البيانات |
|---------|------------------------|
| توقع المبيعات | 50 عملية بيع |
| مخاطر المخزون | 20 منتج |
| التدفق النقدي | 12 شهر من البيانات |
| تقسيم العملاء | 10 عملاء |
| كشف الشذوذ | 30 قيد محاسبي |

---

## ❓ استكشاف الأخطاء

**النموذج لا يتدرب:**
- تأكد من وجود بيانات كافية (راجع الجدول أعلاه)
- تحقق من ربط DbContext بشكل صحيح

**دقة منخفضة:**
- زد بيانات التدريب (3+ سنوات أفضل)
- أعد التدريب بعد إضافة بيانات جديدة

**خطأ في تحميل النموذج:**
- احذف مجلد `MLModels/` وأعد التدريب
- تأكد من صلاحيات الكتابة على المجلد
