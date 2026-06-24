const pptxgen = require("pptxgenjs");

const INK = "2D1B2E";    // тъмен баклажан (тъмни слайдове)
const PLUM = "7B2D5E";   // основен
const ROSE = "C75B8A";   // акцент
const PINK = "F2D7E3";   // светъл тинт
const PAPER = "FFFFFF";
const MUTED = "6B5B6E";
const HEAD = "Georgia";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.author = "Plamena Ilieva";
pres.title = "Разпознаване на медицинска аномалия (CAMELYON17)";

const W = 10, H = 5.625;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.12 });

function title(slide, text, sub) {
  slide.addText(text, { x: 0.6, y: 0.38, w: 8.8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: PLUM, margin: 0 });
  if (sub) slide.addText(sub, { x: 0.62, y: 1.05, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 14, italic: true, color: MUTED, margin: 0 });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || PAPER }, line: { color: PINK, width: 1 }, shadow: shadow() });
}

// ---------- Slide 1: Title ----------
let s = pres.addSlide();
s.background = { color: INK };
// декоративни "клетки"
[[8.7, 0.7, 1.1], [9.3, 1.7, 0.7], [8.4, 2.1, 0.5]].forEach(([cx, cy, d]) =>
  s.addShape(pres.shapes.OVAL, { x: cx, y: cy, w: d, h: d, fill: { color: ROSE, transparency: 55 }, line: { color: PINK, width: 1, transparency: 40 } }));
s.addText("Разпознаване на\nмедицинска аномалия", { x: 0.7, y: 1.5, w: 7.6, h: 1.7, fontFace: HEAD, fontSize: 40, bold: true, color: PAPER, lineSpacingMultiple: 1.0 });
s.addText("Дълбоко самообучение върху CAMELYON17 — засичане на метастази в лимфни възли", { x: 0.72, y: 3.2, w: 7.6, h: 0.8, fontFace: BODY, fontSize: 17, color: PINK });
s.addText([{ text: "Проект 2 · Deep Learning", options: { bold: true } }, { text: "   |   CNN · VGG · U-Net · Streamlit", options: { color: ROSE } }],
  { x: 0.72, y: 4.5, w: 8, h: 0.5, fontFace: BODY, fontSize: 13, color: PINK });

// ---------- Slide 2: Задача ----------
s = pres.addSlide();
title(s, "Задачата");
s.addText([
  { text: "Цел\n", options: { bold: true, fontSize: 18, color: PLUM, breakLine: true } },
  { text: "Уеб приложение, което засича метастази (tumor) в хистопатологични патчове от лимфни възли.\n\n", options: { fontSize: 15, color: INK, breakLine: true } },
  { text: "Защо е „аномалия“\n", options: { bold: true, fontSize: 18, color: PLUM, breakLine: true } },
  { text: "Туморната тъкан е рядка и морфологично различна от нормалната → проблем на anomaly / out-of-distribution detection.", options: { fontSize: 15, color: INK } },
], { x: 0.6, y: 1.4, w: 5.3, h: 3.6, valign: "top", lineSpacingMultiple: 1.05 });

card(s, 6.3, 1.5, 3.1, 3.3, PINK);
s.addText("CAMELYON17", { x: 6.5, y: 1.7, w: 2.7, h: 0.4, fontFace: HEAD, bold: true, fontSize: 16, color: PLUM });
s.addText([
  { text: "1399 whole-slide изображения\n", options: { bullet: true, breakLine: true } },
  { text: "H&E оцветени срезове\n", options: { bullet: true, breakLine: true } },
  { text: "XML анотации на лезиите\n", options: { bullet: true, breakLine: true } },
  { text: "~2.95 TB общо", options: { bullet: true } },
], { x: 6.5, y: 2.2, w: 2.7, h: 2.4, fontFace: BODY, fontSize: 13, color: INK, valign: "top" });

// ---------- Slide 3: Cohen Kappa ----------
s = pres.addSlide();
title(s, "Метрика: Cohen's Kappa", "Съгласуваност, коригирана спрямо случайността");
card(s, 0.6, 1.5, 4.6, 1.3, PAPER);
s.addText([{ text: "κ = ", options: { italic: true } }, { text: "(p", options: {} }, { text: "o", options: { subscript: true } }, { text: " − p", options: {} }, { text: "e", options: { subscript: true } }, { text: ") / (1 − p", options: {} }, { text: "e", options: { subscript: true } }, { text: ")", options: {} }],
  { x: 0.6, y: 1.5, w: 4.6, h: 1.3, align: "center", valign: "middle", fontFace: HEAD, fontSize: 26, bold: true, color: PLUM });
s.addText([
  { text: "pₒ", options: { bold: true } }, { text: " — наблюдавана съгласуваност (accuracy)\n", options: { breakLine: true } },
  { text: "pₑ", options: { bold: true } }, { text: " — очаквана по случайност", options: {} },
], { x: 0.6, y: 2.95, w: 4.6, h: 0.9, fontFace: BODY, fontSize: 13, color: MUTED });
s.addText("Защо тук: датасетът е силно небалансиран → точността подвежда. Kappa коригира спрямо случайността.", { x: 0.6, y: 3.95, w: 4.6, h: 1.0, fontFace: BODY, fontSize: 13, italic: true, color: INK, valign: "top" });

s.addTable([
  [{ text: "Стойност", options: { bold: true, color: PAPER, fill: { color: PLUM } } }, { text: "Тълкуване (Landis & Koch)", options: { bold: true, color: PAPER, fill: { color: PLUM } } }],
  ["< 0", "по-лошо от случайност"],
  ["0.0 – 0.4", "слабо / приемливо"],
  ["0.4 – 0.6", "умерено"],
  ["0.6 – 0.8", "значително"],
  ["0.8 – 1.0", "почти пълно съгласие"],
], { x: 5.5, y: 1.5, w: 3.9, h: 3.2, fontFace: BODY, fontSize: 12.5, color: INK, border: { pt: 0.5, color: PINK }, fill: { color: PAPER }, align: "left", valign: "middle" });

// ---------- Slide 4: Данни / pipeline ----------
s = pres.addSlide();
title(s, "Данни: от слайд до патч");
const steps = [
  ["1", "WSI (.tif)", "Сваляне на под-\nмножество слайдове"],
  ["2", "Tiling 96×96", "Решетка от патчове\nна ниво 2"],
  ["3", "Tissue mask", "Отстраняване на\nфона (HSV наситеност)"],
  ["4", "Етикет", "tumor, ако центърът е\nв анотиран полигон"],
];
steps.forEach(([n, t, d], i) => {
  const x = 0.6 + i * 2.32;
  card(s, x, 1.45, 2.05, 1.85, PAPER);
  s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: 1.6, w: 0.5, h: 0.5, fill: { color: ROSE } });
  s.addText(n, { x: x + 0.15, y: 1.6, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: HEAD, bold: true, fontSize: 18, color: PAPER });
  s.addText(t, { x: x + 0.15, y: 2.18, w: 1.8, h: 0.4, fontFace: HEAD, bold: true, fontSize: 14, color: PLUM });
  s.addText(d, { x: x + 0.15, y: 2.55, w: 1.8, h: 0.7, fontFace: BODY, fontSize: 11.5, color: MUTED });
  if (i < 3) s.addText("→", { x: x + 2.02, y: 1.45, w: 0.34, h: 1.85, align: "center", valign: "middle", fontSize: 20, color: ROSE, bold: true });
});
const stats = [["8000", "патча"], ["96×96×3", "uint8"], ["4000 / 4000", "баланс tumor/normal"], ["8", "пациента"]];
stats.forEach(([v, l], i) => {
  const x = 0.6 + i * 2.32;
  card(s, x, 3.65, 2.05, 1.25, PINK);
  s.addText(v, { x: x + 0.1, y: 3.8, w: 1.85, h: 0.55, align: "center", fontFace: HEAD, bold: true, fontSize: 22, color: PLUM });
  s.addText(l, { x: x + 0.1, y: 4.35, w: 1.85, h: 0.45, align: "center", fontFace: BODY, fontSize: 12, color: MUTED });
});

// ---------- Slide 5: EDA ----------
s = pres.addSlide();
title(s, "Разглеждане на данните (EDA)");
s.addChart(pres.charts.BAR, [
  { name: "mean", labels: ["R", "G", "B"], values: [0.731, 0.607, 0.701] },
  { name: "std", labels: ["R", "G", "B"], values: [0.190, 0.219, 0.167] },
], {
  x: 0.6, y: 1.5, w: 4.8, h: 3.4, barDir: "col", chartColors: [PLUM, ROSE],
  showTitle: true, title: "Пикселна статистика по канал", titleColor: MUTED, titleFontSize: 13,
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 9,
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, valGridLine: { color: "EEE2EA", size: 0.5 },
  showLegend: true, legendPos: "b", legendColor: MUTED, chartArea: { fill: { color: PAPER } },
});
s.addText([
  { text: "Наблюдения\n", options: { bold: true, fontSize: 16, color: PLUM, breakLine: true } },
  { text: "Балансиран набор: 4000 normal / 4000 tumor.\n\n", options: { fontSize: 14, breakLine: true } },
  { text: "R и B са най-високи, G — най-нисък → типично розово-лилаво H&E оцветяване.\n\n", options: { fontSize: 14, breakLine: true } },
  { text: "Вариациите в оцветяването между слайдове са основната пречка за генерализация → адресирани със stain аугментация.", options: { fontSize: 14 } },
], { x: 5.7, y: 1.5, w: 3.7, h: 3.4, fontFace: BODY, color: INK, valign: "top", lineSpacingMultiple: 1.05 });

// ---------- Slide 6: Модели ----------
s = pres.addSlide();
title(s, "Архитектури");
const models = [
  ["SimpleCNN", "Базова CNN", "3 conv блока (Conv-BN-ReLU-Pool) + класификатор. Стъпка 3."],
  ["VGG 11 / 13 / 16", "Дълбоки conv мрежи", "Конфигурируеми VGG блокове с adaptive pooling за 96×96 патчове. Стъпка 4."],
  ["U-Net автоенкодер", "Реконструкция", "Енкодер-декодер със skip връзки; anomaly score = грешка на реконструкцията. Стъпка 5."],
];
models.forEach(([t, sub, d], i) => {
  const x = 0.6 + i * 3.07;
  card(s, x, 1.5, 2.8, 3.3, PAPER);
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w: 2.8, h: 0.12, fill: { color: ROSE }, line: { type: "none" } });
  s.addText(t, { x: x + 0.2, y: 1.8, w: 2.4, h: 0.6, fontFace: HEAD, bold: true, fontSize: 17, color: PLUM });
  s.addText(sub, { x: x + 0.2, y: 2.4, w: 2.4, h: 0.4, fontFace: BODY, italic: true, fontSize: 13, color: ROSE });
  s.addText(d, { x: x + 0.2, y: 2.95, w: 2.45, h: 1.7, fontFace: BODY, fontSize: 13, color: INK, valign: "top" });
});

// ---------- Slide 7: Leakage (★) ----------
s = pres.addSlide();
s.background = { color: INK };
s.addText("Изтичане на информация при оценката", { x: 0.6, y: 0.4, w: 8.8, h: 0.7, fontFace: HEAD, bold: true, fontSize: 28, color: PAPER, margin: 0 });
s.addText("Влияние на начина на разделяне на данните върху резултата", { x: 0.62, y: 1.08, w: 8.8, h: 0.4, fontFace: BODY, italic: true, fontSize: 14, color: ROSE, margin: 0 });
s.addTable([
  [{ text: "Постановка", options: { bold: true, color: PAPER, fill: { color: PLUM } } },
   { text: "CNN", options: { bold: true, color: PAPER, fill: { color: PLUM }, align: "center" } },
   { text: "VGG11", options: { bold: true, color: PAPER, fill: { color: PLUM }, align: "center" } },
   { text: "VGG16", options: { bold: true, color: PAPER, fill: { color: PLUM }, align: "center" } }],
  [{ text: "Наивен split (patch-level)", options: { color: INK } }, { text: "0.98", options: { align: "center", color: "B00020" } }, { text: "—", options: { align: "center" } }, { text: "0.99", options: { align: "center", color: "B00020" } }],
  [{ text: "Patient-level split", options: { color: INK } }, { text: "−0.05", options: { align: "center" } }, { text: "0.79", options: { align: "center" } }, { text: "0.62", options: { align: "center" } }],
  [{ text: "+ ColorJitter + best-epoch", options: { color: INK, bold: true, fill: { color: PINK } } }, { text: "0.87", options: { align: "center", bold: true, fill: { color: PINK } } }, { text: "0.92", options: { align: "center", bold: true, fill: { color: PINK } } }, { text: "0.91", options: { align: "center", bold: true, fill: { color: PINK } } }],
], { x: 0.6, y: 1.7, w: 8.8, h: 2.2, fontFace: BODY, fontSize: 14, color: INK, fill: { color: PAPER }, border: { pt: 0.5, color: "FFFFFF" }, valign: "middle" });
s.addText([
  { text: "Наивният split бърка патчове от един слайд в train и val → моделът „познава слайда“ по оцветяването → фалшиви ~0.98.  ", options: {} },
  { text: "Patient-level split + stain аугментация разкриват честната генерализация.", options: { bold: true, color: PINK } },
], { x: 0.6, y: 4.05, w: 8.8, h: 1.1, fontFace: BODY, fontSize: 13, color: PAPER, valign: "top", lineSpacingMultiple: 1.05 });

// ---------- Slide 8: Резултати ----------
s = pres.addSlide();
title(s, "Резултати — невиждани пациенти", "Cohen Kappa · 30 пациента · 44 321 патча · patient-level split");
s.addChart(pres.charts.BAR, [
  { name: "Cohen Kappa", labels: ["SimpleCNN", "VGG11", "VGG16"], values: [0.725, 0.797, 0.724] },
], {
  x: 0.6, y: 1.5, w: 5.6, h: 3.4, barDir: "col", chartColors: [PLUM, ROSE, PLUM],
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 12, dataLabelFormatCode: "0.000",
  valAxisMinVal: 0, valAxisMaxVal: 1, catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  valGridLine: { color: "EEE2EA", size: 0.5 }, showLegend: false, chartArea: { fill: { color: PAPER } },
});
card(s, 6.5, 1.7, 2.9, 1.5, PINK);
s.addText([{ text: "VGG11\n", options: { bold: true, fontSize: 20, color: PLUM, breakLine: true } }, { text: "най-висок κ = 0.797", options: { fontSize: 14, color: INK } }],
  { x: 6.6, y: 1.85, w: 2.7, h: 1.2, align: "center", valign: "middle", fontFace: HEAD });
s.addText([
  { text: "VGG16 < VGG11 → повече параметри = повече overfitting при малко пациенти.\n\n", options: { breakLine: true } },
  { text: "CNN се вдигна от случайност (−0.05) до 0.73 след stain аугментация.", options: {} },
], { x: 6.5, y: 3.4, w: 2.9, h: 1.6, fontFace: BODY, fontSize: 12.5, color: INK, valign: "top", lineSpacingMultiple: 1.05 });

// ---------- Slide 9: U-Net negative ----------
s = pres.addSlide();
title(s, "Автоенкодер: честен отрицателен резултат");
card(s, 0.6, 1.5, 4.3, 2.0, PAPER);
s.addText([
  { text: "recon MSE\n", options: { bold: true, fontSize: 15, color: PLUM, breakLine: true } },
  { text: "normal: 0.0006\n", options: { fontSize: 18, breakLine: true } },
  { text: "tumor:  0.0004\n", options: { fontSize: 18, breakLine: true } },
  { text: "ROC AUC < 0.5  (обърнат score)", options: { fontSize: 14, italic: true, color: ROSE } },
], { x: 0.8, y: 1.65, w: 3.9, h: 1.7, fontFace: BODY, color: INK, valign: "top" });
s.addText([
  { text: "Защо не работи\n", options: { bold: true, fontSize: 16, color: PLUM, breakLine: true } },
  { text: "Туморните региони се реконструират по-добре от нормалните. Нормалната лимфна тъкан е структурно по-богата (лимфоцити, граници) → по-трудна за реконструкция; туморът е по-хомогенен → по-лесен.\n\n", options: { fontSize: 14, breakLine: true } },
  { text: "Извод: за тези данни supervised CNN/VGG далеч превъзхожда reconstruction-based подхода.", options: { fontSize: 14, bold: true, color: PLUM } },
], { x: 5.2, y: 1.5, w: 4.2, h: 3.4, fontFace: BODY, color: INK, valign: "top", lineSpacingMultiple: 1.05 });

// ---------- Slide 10: Инженерни находки ----------
s = pres.addSlide();
title(s, "Инженерна стриктност", "Бъгове, открити и поправени по пътя (BDD тестове: 25 ✓)");
const bugs = [
  ["Само-normal извличане", "Последователното сканиране с лимит пропускаше малките лезии → 0 tumor патча. Решение: насочено семплиране от полигоните."],
  ["Ред на трансформите", "Аугментациите се прилагаха върху numpy масив. Решение: ToTensor преди тях."],
  ["float32 средни", "Каналните средни колабираха при ~74 млн пиксела. Решение: float64 акумулатор."],
];
bugs.forEach(([t, d], i) => {
  const y = 1.55 + i * 1.15;
  card(s, 0.6, y, 8.8, 1.0, PAPER);
  s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.28, w: 0.45, h: 0.45, fill: { color: ROSE } });
  s.addText(String(i + 1), { x: 0.8, y: y + 0.28, w: 0.45, h: 0.45, align: "center", valign: "middle", fontFace: HEAD, bold: true, fontSize: 16, color: PAPER });
  s.addText([{ text: t + "  —  ", options: { bold: true, color: PLUM } }, { text: d, options: { color: INK } }],
    { x: 1.45, y: y + 0.12, w: 7.8, h: 0.78, fontFace: BODY, fontSize: 13, valign: "middle" });
});

// ---------- Slide 11: Демо ----------
s = pres.addSlide();
title(s, "Потребителски интерфейс (Streamlit)");
card(s, 0.6, 1.5, 4.6, 3.3, PAPER);
s.addText("Качване на патч и предсказание", { x: 0.8, y: 1.7, w: 4.2, h: 0.4, fontFace: HEAD, bold: true, fontSize: 15, color: PLUM });
s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 2.2, w: 1.6, h: 1.6, fill: { color: PINK }, line: { color: ROSE, width: 1 } });
s.addText("патч", { x: 0.9, y: 2.2, w: 1.6, h: 1.6, align: "center", valign: "middle", fontFace: BODY, fontSize: 12, color: MUTED });
s.addText([
  { text: "tumor (метастаза)\n", options: { bold: true, fontSize: 15, color: "B00020", breakLine: true } },
  { text: "вероятност: 91%", options: { fontSize: 13, color: INK } },
], { x: 2.7, y: 2.5, w: 2.4, h: 1.0, fontFace: BODY, valign: "top" });
s.addShape(pres.shapes.RECTANGLE, { x: 2.7, y: 3.5, w: 2.3, h: 0.22, fill: { color: PINK } });
s.addShape(pres.shapes.RECTANGLE, { x: 2.7, y: 3.5, w: 2.09, h: 0.22, fill: { color: ROSE } });
s.addText([
  { text: "Възможности\n", options: { bold: true, fontSize: 16, color: PLUM, breakLine: true } },
  { text: "Качване на изображение, предсказание tumor/normal + вероятност.\n\n", options: { bullet: true, fontSize: 14, breakLine: true } },
  { text: "Зарежда архитектурата автоматично от чекпойнта.\n\n", options: { bullet: true, fontSize: 14, breakLine: true } },
  { text: "Логиката е изнесена и покрита с BDD тестове.", options: { bullet: true, fontSize: 14 } },
], { x: 5.5, y: 1.6, w: 3.9, h: 3.2, fontFace: BODY, color: INK, valign: "top", lineSpacingMultiple: 1.05 });

// ---------- Slide 12: Изводи ----------
s = pres.addSlide();
s.background = { color: INK };
s.addText("Изводи", { x: 0.6, y: 0.5, w: 8.8, h: 0.8, fontFace: HEAD, bold: true, fontSize: 32, color: PAPER, margin: 0 });
s.addText([
  { text: "VGG11 е най-добрият модел (Cohen Kappa 0.797 на 30 невиждани пациента).\n", options: { bullet: true, breakLine: true } },
  { text: "Stain аугментацията е критична за генерализация в хистопатологията.\n", options: { bullet: true, breakLine: true } },
  { text: "Patient-level евалуация е задължителна — иначе резултатите са фалшиво високи.\n", options: { bullet: true, breakLine: true } },
  { text: "Reconstruction-based anomaly detection не работи за тези данни (честен отрицателен резултат).", options: { bullet: true } },
], { x: 0.7, y: 1.6, w: 8.7, h: 2.2, fontFace: BODY, fontSize: 16, color: PINK, valign: "top", lineSpacingMultiple: 1.15 });
s.addText([
  { text: "Бъдеща работа:  ", options: { bold: true, color: ROSE } },
  { text: "повече пациенти · отделен test сет · transfer learning · U-Net сегментация на tumor маска.", options: { color: PINK } },
], { x: 0.7, y: 4.5, w: 8.7, h: 0.7, fontFace: BODY, fontSize: 14, valign: "top" });

pres.writeFile({ fileName: "../docs/presentation.pptx" }).then(f => console.log("written", f));
