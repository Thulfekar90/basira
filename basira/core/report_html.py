"""Self-contained HTML forensic report.

One file, no network, readable with JavaScript disabled, printable, English or Arabic.

Rules this module follows
-------------------------
* **Nothing external.** CSS, JS, SVG and images are inline; a strict meta
  Content-Security-Policy forbids every network request and only lets the one
  static script below run (by hash), so an escaping bug cannot become script
  execution.
* **Everything from the analysed file is hostile**: file names, metadata values,
  detector summaries quoting metadata. All of it goes through :func:`_esc`
  (HTML text and attributes) or :func:`_json_for_script` (the JSON block).
* **JavaScript only enhances.** Layer switches are pure CSS; opacity, the
  before/after wipe, zoom, the lightbox and plot read-outs need the script.
* **Interface text lives in** :data:`STRINGS` **and** :data:`DETECTORS`.
  Detector summaries are written in English by the engines and are shown as
  they are, in ``dir="ltr"`` blocks, rather than machine-translated.

Public API: :func:`render_html`, :func:`save_html`, :data:`CSS` (shared with the
upload page in ``basira/server.py``).
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import html
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .types import Evidence, Report, Segment
from .utils import colormap, thumbnail

LANGS = ("en", "ar")

# Size budget (see tests/test_report.py): a report stays under ~1.5 MB whatever the content.
# The analysed picture is embedded once as a JPEG; maps are small indexed PNGs laid over
# it; every JPEG has a byte cap and is re-encoded smaller until it fits.
BASE_MAX_SIDE = 1600        # analysed image / key frame, embedded once
BASE_QUALITY = 82
BASE_MAX_BYTES = 540_000
VISUAL_MAX_SIDE = 720       # detector visuals (ELA, match lines, spectrum ...)
VISUAL_QUALITY = 76
VISUAL_MAX_BYTES = 100_000  # each
VISUALS_TOTAL_BYTES = 330_000   # all visuals of one report, child reports included
HEAT_MAX_SIDE = 400         # localisation maps are smooth; the browser upscales them
HEAT_LEVELS = 64
PLOT_MAX_POINTS = 800
RAW_JSON_LIMIT = 8000       # characters of "raw measurements" per detector
MAX_LAYERS = 10             # CSS below defines switches for this many map layers

# Categorical colours for per-detector map layers and boxes. A detector keeps its
# colour in every report; detectors not listed take the next free slot.
LAYER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6",
                "#e34948", "#008300")
LAYER_SLOT = {"image.jpeg_dq": 0, "image.copymove": 1, "video.frame_copymove": 1,
              "image.noise": 2, "image.cfa": 3, "image.ela": 4, "image.thumbnail": 5}

# --------------------------------------------------------------------------- #
# Interface text
# --------------------------------------------------------------------------- #
STRINGS: Dict[str, Dict[str, Any]] = {
    "en": {
        "title": "Basira forensic report",
        "report": "Forensic report",
        "skip": "Skip to the verdict",
        "print": "Print / save as PDF",
        "close": "Close",
        "enlarge": "Enlarge",
        "nav": "Sections of this report",
        # file facts
        "facts": "File under analysis",
        "file": "File", "sha256": "SHA-256", "size": "Size", "type": "Type",
        "generated": "Report generated", "tool": "Tool", "elapsed": "Analysis time",
        "sha_missing": "not available: the file was no longer on disk when this report was written",
        "bytes": "bytes", "seconds": "s",
        "media": {"image": "Image", "audio": "Audio", "video": "Video"},
        # verdict
        "verdict": "Verdict",
        "scale": "Position on Basira's five-step scale",
        "steps": ["No traces", "Inconclusive", "Suspicious", "Likely", "Provenance evidence"],
        "means": "What this means",
        "not_means": "What it does not mean",
        "scores": "Category scores",
        "scores_help": "Fused evidence per category. The marks show where the verdict changes: "
                       "20% inconclusive, 45% suspicious, 75% likely.",
        "prov_help": "Provenance flags describe the file's history. On their own they never "
                     "raise the verdict above “inconclusive” unless the metadata is decisive.",
        "cat": {"manipulation": "Manipulation", "ai_generation": "AI generation",
                "provenance": "Provenance"},
        "cat_score": {"manipulation": "Manipulation", "ai_generation": "AI generation",
                      "provenance": "Provenance flags"},
        "findings": "Main findings",
        "findings_note": "In the detectors' own words.",
        "track": "audio track",
        # viewer
        "where_image": "Where in the image",
        "where_video": "Key frame",
        "keyframe_note": "One key frame is shown for reference. Findings in time are on the "
                         "timeline below.",
        "layers": "Map layers",
        "fused": "Fused localisation map",
        "fused_legend": "low to high",
        "boxes": "Region boxes",
        "below": "below threshold, not evidence",
        "opacity": "Map opacity",
        "compare": "Compare with the original",
        "compare_hint": "Slide to wipe the maps away from the left.",
        "zoom": "Zoom",
        "zoom_hint": "Click the image to zoom in, click again to zoom out.",
        "scroll_label": "Analysed image with localisation maps. Scroll to pan when zoomed.",
        "no_map": "No localisation map: no detector with a spatial map crossed its threshold.",
        "regions": "Marked regions",
        "region_words": "{pos} of the image, {w}×{h} px at ({x}, {y}), about {pct}% of the area",
        "region_frame": "frame {n}",
        "pos": [["top left", "top centre", "top right"],
                ["centre left", "centre", "centre right"],
                ["bottom left", "bottom centre", "bottom right"]],
        "map_alt": "Fused localisation map: the highlighted area covers about {pct}% of the "
                   "image, centred on the {pos}.",
        "map_alt_layer": "Map of {name}: the highlighted area covers about {pct}% of the image, "
                         "centred on the {pos}.",
        "map_alt_none": "Map of {name}: nothing is strongly highlighted.",
        "alt_image": "The analysed image",
        "alt_video": "Key frame of the analysed video",
        "alt_audio": "Spectrogram of the recording: time runs left to right, frequency bottom "
                     "to top; bands flagged by the detectors are tinted",
        # timeline
        "timeline": "Timeline",
        "spectrogram": "Spectrogram",
        "lane_all": "All detectors, merged",
        "ranges": "Suspicious time ranges",
        "no_ranges": "No suspicious time ranges were reported.",
        "tl_alt": "Timeline of {dur}: {n} suspicious range(s). They are listed in the table "
                  "that follows.",
        "col_from": "From", "col_to": "To", "col_score": "Score", "col_what": "What was measured",
        "duration": "Duration",
        # evidence
        "evidence": "Evidence",
        "evidence_help": "Every detector that was run, grouped by outcome. A detector's score is "
                         "0.5 exactly at its own decision threshold; only the part above 0.5 "
                         "counts, scaled by the detector's confidence on this file.",
        "overview": "All detectors at a glance",
        "col_detector": "Detector", "col_category": "Category", "col_outcome": "Outcome",
        "col_conf": "Confidence", "col_finding": "Finding",
        "state": {"fired": "Fired", "quiet": "Quiet", "na": "Not applicable", "failed": "Failed"},
        "state_help": {
            "fired": "crossed their threshold on this file",
            "quiet": "ran and stayed below their threshold",
            "na": "could not be applied to this file",
            "failed": "crashed or could not run, so their silence means nothing"},
        "score": "Score", "confidence": "Confidence", "weight": "Weight in the verdict",
        "threshold": "threshold",
        "decisive": "Decisive",
        "low_conf": "low confidence on this file",
        "measures": "Measures",
        "blind": "Blind spot",
        "no_about": "No description is available for this detector in this version of the report.",
        "finding": "Finding",
        "error": "Error",
        "raw": "Raw measurements",
        "truncated": "(truncated)",
        "show_on_image": "Show this map on the image",
        "plot_alt": "Plot of {y} against {x}, from {x0} to {x1}; values between {y0} and {y1}.",
        "plot_marks": "Marked at {list}.",
        "marked": "marked",
        # how the verdict was reached (evidence flow diagram)
        "flow": "How the verdict was reached",
        "flow_help": "The real path of this analysis, drawn from the measurements above. Each "
                     "detector's line is as thick as its weight in the verdict: the part of its "
                     "score above its threshold, times its confidence on this file. Detectors "
                     "below their threshold add nothing.",
        "flow_replay": "Replay the analysis",
        "flow_cats": "Categories",
        "flow_s1": "The file was read",
        "flow_s2": "{total} detectors were applied: {fired} crossed their threshold, {quiet} "
                   "stayed below it, {na} did not apply and {failed} failed.",
        "flow_s2_track": "The embedded sound track was analysed separately and feeds the result.",
        "flow_s3": "Fused per category: manipulation {m}%, AI generation {a}%, provenance "
                   "flags {p}%.",
        "flow_s4": "The verdict follows the stronger category, {cat} at {v}%, against the marks "
                   "at 20%, 45% and 75%.",
        "flow_s4_dec": "Decisive provenance evidence settles the verdict without the marks.",
        "flow_s5": "Verdict",
        # file details, children, footer
        "details": "File details",
        "details_help": "Technical properties as read by the tool.",
        "child": "Embedded {kind} track",
        "child_note": "The sound track was extracted and analysed by the audio engine. Its "
                      "findings feed the verdict at the top of this report.",
        "howto": "How to read this report",
        "disclaimer": "Basira reports evidence, not proof. It never certifies that a file is "
                      "authentic: the best possible result is “No significant traces found”, "
                      "because re-compression, resizing and careful editing erase the traces its "
                      "detectors rely on. Treat this report as one input to human judgement, "
                      "never as the sole basis for an accusation.",
        "offline": "This file is self-contained: it loads nothing from the network.",
        # verdict texts: (label or None to use the engine's own wording, means, does not mean)
        "v": {
            "v0": (None,
                   "None of the detectors that could run on this file crossed its threshold.",
                   "It does not mean the file is authentic or unedited. Re-compression, "
                   "resizing, screenshots and careful editing remove the traces these detectors "
                   "rely on, and they miss more than they catch."),
            "v1": (None,
                   "Weak anomalies were measured, below the level at which Basira calls a file "
                   "suspicious.",
                   "It is neither a finding of manipulation nor a clean result, and should not "
                   "be reported as either."),
            "v1_unreadable": (None,
                              "The content could not be decoded, so no content detector ran.",
                              "The absence of findings means nothing here. This is not a clean "
                              "result."),
            "v1_partial": (None,
                           "Part of the analysis failed. The detectors that did run found "
                           "nothing.",
                           "It is not a clean result: the analysis is incomplete."),
            "v2": (None,
                   "At least one detector measured an anomaly above its threshold. It is a lead "
                   "that deserves examination by a person.",
                   "It is not proof of manipulation. Detectors raise false alarms, and some "
                   "ordinary processing leaves the same traces."),
            "v3_manip": (None,
                         "Strong or mutually reinforcing traces consistent with editing were "
                         "measured. The marked regions or time ranges show where.",
                         "It is not proof, and it says nothing about who changed the file, why, "
                         "or whether the change misleads. A legitimate edit leaves the same "
                         "traces."),
            "v3_ai": (None,
                      "Strong or mutually reinforcing traces consistent with synthetic content "
                      "were measured.",
                      "It is not proof. Without provenance data this is the weakest question "
                      "Basira answers: upsampled, transcoded or noise-gated material can "
                      "produce the same cues."),
            "v4_ai": (None,
                      "The file itself carries metadata written by an AI generator, or an AI "
                      "label (C2PA / IPTC). The verdict rests on that statement, not on a "
                      "measurement of the content.",
                      "It does not show how much of the content is synthetic. Metadata can be "
                      "removed, or written into a file by anyone: treat it as a strong lead "
                      "about origin."),
            "v4_edit": (None,
                        "The file itself carries decisive metadata about how it was produced. "
                        "The verdict rests on that statement, not on a measurement of the "
                        "content.",
                        "It does not show what was changed. Metadata can be removed, or written "
                        "into a file by anyone: treat it as a strong lead about origin."),
        },
        # sentences written by the fusion step (English only); translated where known
        "fusion": {},
    },
    "ar": {
        "title": "تقرير بصيرة للفحص الجنائي الرقمي",
        "report": "تقرير الفحص الجنائي الرقمي",
        "skip": "انتقل إلى النتيجة",
        "print": "طباعة / حفظ بصيغة PDF",
        "close": "إغلاق",
        "enlarge": "تكبير",
        "nav": "أقسام التقرير",
        "facts": "الملف قيد الفحص",
        "file": "الملف", "sha256": "بصمة SHA-256", "size": "الحجم", "type": "النوع",
        "generated": "تاريخ إنشاء التقرير", "tool": "الأداة", "elapsed": "مدة التحليل",
        "sha_missing": "غير متاحة: لم يعد الملف موجودًا على القرص عند كتابة هذا التقرير",
        "bytes": "بايت", "seconds": "ث",
        "media": {"image": "صورة", "audio": "صوت", "video": "فيديو"},
        "verdict": "النتيجة",
        "scale": "موقع النتيجة على سلّم بصيرة ذي الدرجات الخمس",
        "steps": ["لا آثار", "غير حاسم", "مثير للشك", "مرجَّح", "دليل من بيانات الملف"],
        "means": "ماذا تعني هذه النتيجة",
        "not_means": "ما الذي لا تعنيه",
        "scores": "درجات الفئات",
        "scores_help": "حصيلة الأدلة في كل فئة. العلامات تبيّن أين تتغيّر النتيجة: 20% غير حاسم، "
                       "45% مثير للشك، 75% مرجَّح.",
        "prov_help": "مؤشرات المصدر تصف تاريخ الملف. وهي وحدها لا ترفع النتيجة فوق «غير حاسم» "
                     "ما لم تكن البيانات الوصفية قاطعة.",
        "cat": {"manipulation": "تلاعب", "ai_generation": "توليد بالذكاء الاصطناعي",
                "provenance": "المصدر"},
        "cat_score": {"manipulation": "التلاعب", "ai_generation": "التوليد بالذكاء الاصطناعي",
                      "provenance": "مؤشرات المصدر"},
        "findings": "أبرز ما وُجد",
        "findings_note": "بعبارات الكواشف نفسها (بالإنجليزية).",
        "track": "المسار الصوتي",
        "where_image": "أين في الصورة",
        "where_video": "إطار مفتاحي",
        "keyframe_note": "يُعرض إطار مفتاحي واحد للاستئناس. أما ما وُجد على امتداد الزمن ففي "
                         "الخط الزمني أدناه.",
        "layers": "طبقات الخرائط",
        "fused": "خريطة التحديد المدمجة",
        "fused_legend": "من الأضعف إلى الأقوى",
        "boxes": "إطارات المناطق",
        "below": "دون العتبة، وليست دليلًا",
        "opacity": "شفافية الخريطة",
        "compare": "المقارنة بالأصل",
        "compare_hint": "اسحب لإزاحة الخرائط من جهة اليسار.",
        "zoom": "تكبير",
        "zoom_hint": "انقر على الصورة للتكبير، وانقر مرة أخرى للتصغير.",
        "scroll_label": "الصورة المفحوصة مع خرائط التحديد. مرِّر للتنقل عند التكبير.",
        "no_map": "لا توجد خريطة تحديد: لم يتجاوز أي كاشف مكاني عتبته.",
        "regions": "المناطق المحدَّدة",
        "region_words": "في {pos} الصورة، {w}×{h} بكسل عند ({x}، {y})، نحو {pct}% من المساحة",
        "region_frame": "الإطار {n}",
        "pos": [["أعلى يسار", "أعلى وسط", "أعلى يمين"],
                ["وسط يسار", "وسط", "وسط يمين"],
                ["أسفل يسار", "أسفل وسط", "أسفل يمين"]],
        "map_alt": "خريطة التحديد المدمجة: المنطقة المظلَّلة تغطي نحو {pct}% من الصورة، "
                   "ومركزها في {pos} الصورة.",
        "map_alt_layer": "خريطة {name}: المنطقة المظلَّلة تغطي نحو {pct}% من الصورة، ومركزها "
                         "في {pos} الصورة.",
        "map_alt_none": "خريطة {name}: لا توجد منطقة مظلَّلة بوضوح.",
        "alt_image": "الصورة المفحوصة",
        "alt_video": "إطار مفتاحي من الفيديو المفحوص",
        "alt_audio": "المخطط الطيفي للتسجيل: الزمن من اليسار إلى اليمين، والتردد من الأسفل إلى "
                     "الأعلى، والمقاطع التي أشارت إليها الكواشف ملوَّنة",
        "timeline": "الخط الزمني",
        "spectrogram": "المخطط الطيفي",
        "lane_all": "كل الكواشف مجتمعة",
        "ranges": "المقاطع الزمنية المشبوهة",
        "no_ranges": "لم يُبلَّغ عن مقاطع زمنية مشبوهة.",
        "tl_alt": "خط زمني مدته {dur}: فيه {n} من المقاطع المشبوهة، وهي مسرودة في الجدول التالي.",
        "col_from": "من", "col_to": "إلى", "col_score": "الدرجة", "col_what": "ما الذي قيس",
        "duration": "المدة",
        "evidence": "الأدلة",
        "evidence_help": "كل كاشف جرى تشغيله، مصنَّفًا بحسب ما انتهى إليه. درجة الكاشف تساوي 0.5 "
                         "عند عتبة قراره تمامًا؛ ولا يُحتسب إلا ما زاد على 0.5، مضروبًا في ثقة "
                         "الكاشف بهذا الملف.",
        "overview": "نظرة سريعة على كل الكواشف",
        "col_detector": "الكاشف", "col_category": "الفئة", "col_outcome": "الحالة",
        "col_conf": "الثقة", "col_finding": "ما وُجد",
        "state": {"fired": "تجاوز العتبة", "quiet": "دون العتبة", "na": "لا ينطبق",
                  "failed": "فشل"},
        "state_help": {
            "fired": "تجاوزت عتبتها في هذا الملف",
            "quiet": "عملت وبقيت دون عتبتها",
            "na": "تعذّر تطبيقها على هذا الملف",
            "failed": "تعطّلت أو لم تتمكن من العمل، فسكوتها لا يعني شيئًا"},
        "score": "الدرجة", "confidence": "الثقة", "weight": "الوزن في النتيجة",
        "threshold": "العتبة",
        "decisive": "قاطع",
        "low_conf": "ثقة منخفضة في هذا الملف",
        "measures": "ما يقيسه",
        "blind": "ما لا يراه",
        "no_about": "لا يتوفر وصف لهذا الكاشف في هذه النسخة من التقرير.",
        "finding": "ما وُجد",
        "error": "الخطأ",
        "raw": "القياسات الخام",
        "truncated": "(مقتطع)",
        "show_on_image": "اعرض هذه الخريطة على الصورة",
        "plot_alt": "رسم بياني لـ{y} مقابل {x}، من {x0} إلى {x1}؛ والقيم بين {y0} و{y1}.",
        "plot_marks": "مواضع معلَّمة عند {list}.",
        "marked": "مواضع معلَّمة",
        "flow": "كيف وصل التقرير إلى النتيجة",
        "flow_help": "المسار الفعلي لهذا التحليل، مرسومًا من القياسات نفسها. سماكة خط كل كاشف "
                     "بقدر وزنه في النتيجة: ما زاد من درجته على عتبته، مضروبًا في ثقته بهذا "
                     "الملف. والكواشف التي بقيت دون عتبتها لا تضيف شيئًا.",
        "flow_replay": "أعد عرض التحليل",
        "flow_cats": "الفئات",
        "flow_s1": "قُرئ الملف",
        "flow_s2": "طُبّق {total} من الكواشف: تجاوز عتبتَه منها {fired}، وبقي دونها {quiet}، "
                   "ولم ينطبق {na}، وفشل {failed}.",
        "flow_s2_track": "حُلّل المسار الصوتي المضمَّن على حدة، ويدخل ما وُجد فيه في النتيجة.",
        "flow_s3": "حصيلة كل فئة: التلاعب {m}%، والتوليد بالذكاء الاصطناعي {a}%، ومؤشرات المصدر "
                   "{p}%.",
        "flow_s4": "تتبع النتيجة الفئة الأقوى، {cat} بنسبة {v}%، مقارنةً بالعلامات 20% و45% "
                   "و75%.",
        "flow_s4_dec": "دليل قاطع من بيانات الملف يحسم النتيجة دون الرجوع إلى العلامات.",
        "flow_s5": "النتيجة",
        "details": "تفاصيل الملف",
        "details_help": "الخصائص التقنية كما قرأتها الأداة.",
        "child": "المسار المضمَّن: {kind}",
        "child_note": "استُخرج المسار الصوتي وحلّله محرّك الصوت. وما وُجد فيه يدخل في النتيجة "
                      "الواردة أعلى التقرير.",
        "howto": "كيف يُقرأ هذا التقرير",
        "disclaimer": "بصيرة تقدّم قرائن لا براهين. وهي لا تشهد أبدًا بأن الملف أصلي: أفضل نتيجة "
                      "ممكنة هي «لم يُعثر على آثار تُذكر»، لأن إعادة الضغط وتغيير الحجم والتعديل "
                      "المتقن تمحو الآثار التي تعتمد عليها الكواشف. فليكن هذا التقرير عنصرًا "
                      "واحدًا بين يدي من يحكم، لا الأساس الوحيد لأي اتهام.",
        "offline": "هذا الملف مكتفٍ بذاته: لا يحمّل شيئًا من الشبكة.",
        "v": {
            "v0": ("لم يُعثر على آثار تُذكر",
                   "لم يتجاوز عتبتَه أيٌّ من الكواشف التي أمكن تشغيلها على هذا الملف.",
                   "لا تعني أن الملف أصلي أو غير معدَّل. فإعادة الضغط وتغيير الحجم ولقطات الشاشة "
                   "والتعديل المتقن تمحو الآثار التي تعتمد عليها هذه الكواشف، وما يفوتها أكثر "
                   "مما تلتقطه."),
            "v1": ("غير حاسم (مؤشرات ضعيفة)",
                   "قيست مؤشرات ضعيفة، دون المستوى الذي تعدّ بصيرة عنده الملف مثيرًا للشك.",
                   "ليست دليلًا على التلاعب ولا شهادة بالسلامة، ولا يصحّ عرضها على أنها هذا "
                   "أو ذاك."),
            "v1_unreadable": ("غير حاسم (تعذّر تحليل المحتوى)",
                              "تعذّر فكّ ترميز المحتوى، فلم يعمل أي كاشف للمحتوى.",
                              "غياب المؤشرات لا يعني شيئًا هنا، وهذه ليست نتيجة سليمة."),
            "v1_partial": ("غير حاسم (فشل جزء من التحليل)",
                           "فشل جزء من التحليل. والكواشف التي عملت لم تجد شيئًا.",
                           "ليست نتيجة سليمة: فالتحليل ناقص."),
            "v2": ("مثير للشك",
                   "قاس كاشف واحد على الأقل شذوذًا يتجاوز عتبته. وهذا خيط يستحق أن يفحصه مختص.",
                   "ليست دليلًا قاطعًا على التلاعب. فالكواشف تخطئ أحيانًا، وبعض المعالجات "
                   "العادية تترك الآثار نفسها."),
            "v3_manip": ("يُرجَّح أنه تعرّض للتلاعب",
                         "قيست آثار قوية، أو آثار يعضد بعضها بعضًا، تتوافق مع التعديل. والمناطق "
                         "أو المقاطع المحدَّدة تبيّن موضعها.",
                         "ليست برهانًا، ولا تقول شيئًا عمّن غيّر الملف ولا عن سبب ذلك ولا عمّا إذا "
                         "كان التغيير مضلِّلًا. فالتعديل المشروع يترك الآثار نفسها."),
            "v3_ai": ("يُرجَّح أنه مولَّد بالذكاء الاصطناعي",
                      "قيست آثار قوية، أو آثار يعضد بعضها بعضًا، تتوافق مع محتوى اصطناعي.",
                      "ليست برهانًا. ففي غياب بيانات المصدر يكون هذا أضعف سؤال تجيب عنه بصيرة: "
                      "المواد التي رُفع معدل عيّناتها أو أُعيد ترميزها أو عولجت ببوابة ضجيج قد "
                      "تعطي القرائن نفسها."),
            "v4_ai": ("مولَّد بالذكاء الاصطناعي (بدليل من بيانات الملف)",
                      "الملف نفسه يحمل بيانات وصفية كتبها مولِّد ذكاء اصطناعي، أو وسمًا يصرّح "
                      "بذلك (C2PA / IPTC). فالنتيجة تستند إلى هذا التصريح لا إلى قياس للمحتوى.",
                      "لا تبيّن مقدار ما هو اصطناعي من المحتوى. والبيانات الوصفية يمكن حذفها، "
                      "ويمكن لأي شخص أن يكتبها في ملف: فهي خيط قوي عن المنشأ لا أكثر."),
            "v4_edit": ("معدَّل (بدليل من بيانات الملف)",
                        "الملف نفسه يحمل بيانات وصفية قاطعة عن طريقة إنتاجه. فالنتيجة تستند إلى "
                        "هذا التصريح لا إلى قياس للمحتوى.",
                        "لا تبيّن ما الذي تغيّر. والبيانات الوصفية يمكن حذفها، ويمكن لأي شخص أن "
                        "يكتبها في ملف: فهي خيط قوي عن المنشأ لا أكثر."),
        },
        "fusion": {
            "No detector exceeded its threshold. This is not proof of authenticity: skilled "
            "edits and strong re-compression can erase the traces these detectors rely on.":
                "لم يتجاوز أي كاشف عتبته. وهذا ليس دليلًا على الأصالة: فالتعديل المتقن وإعادة "
                "الضغط الشديدة قد يمحوان الآثار التي تعتمد عليها هذه الكواشف.",
            "No content detector could run on this file, so the absence of findings means "
            "nothing.":
                "لم يتمكن أي كاشف للمحتوى من العمل على هذا الملف، فغياب المؤشرات لا يعني شيئًا.",
            "The remaining detectors found nothing, but the analysis is incomplete.":
                "لم تجد بقية الكواشف شيئًا، لكن التحليل ناقص.",
            "part of its analysis failed": "فشل جزء من تحليله",
        },
    },
}

# What each detector measures and its main blind spot. Facts are taken from
# docs/METHODS.md and the "Limitations" section of the README; nothing is claimed here
# that is not stated there or in the detector's own output. "ar" adds an Arabic name.
DETECTORS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "image.jpeg_dq": {
        "en": ("Looks for the comb pattern that a second JPEG save leaves in the DCT "
               "coefficient histograms, and maps the blocks that do not share it.",
               "Needs a JPEG saved at least twice with its 8×8 grid intact; one more lossy "
               "save or a resize erases the trace."),
        "ar": ("خريطة الضغط المزدوج في JPEG",
               "يبحث عن النمط المشطي الذي يتركه الحفظ الثاني بصيغة JPEG في توزيع معاملات DCT، "
               "ويحدّد المربعات التي لا تحمل هذا الأثر.",
               "يتطلب صورة JPEG حُفظت مرتين على الأقل مع بقاء شبكة 8×8 سليمة؛ وأي حفظ إضافي "
               "بضغط أو تغيير للحجم يمحو الأثر.")},
    "image.copymove": {
        "en": ("Matches keypoints inside the image and accepts a clone only if the fine detail "
               "of the two areas (the noise residual) also matches.",
               "Needs texture, rejects most rotated or rescaled clones, and cannot tell a "
               "digital clone from identical printed objects."),
        "ar": ("النسخ واللصق داخل الصورة",
               "يطابق النقاط المميِّزة داخل الصورة نفسها، ولا يعدّ المنطقتين نسخة إلا إذا تطابقت "
               "تفاصيلهما الدقيقة (بقايا الضجيج) أيضًا.",
               "يحتاج إلى محتوى غني بالتفاصيل، ويرفض معظم النسخ التي دُوِّرت أو غُيِّر حجمها، ولا "
               "يميّز النسخ الرقمي من أجسام مطبوعة متطابقة.")},
    "image.cfa": {
        "en": ("Checks that the camera's demosaicing pattern sits on the same pixel lattice "
               "everywhere; a cluster on the opposite lattice comes from another image.",
               "Only for uncompressed or very high quality images that were not resized; a "
               "region pasted with the same Bayer phase is invisible, so about half of splices "
               "are missed by construction."),
        "ar": ("اتساق نمط مصفوفة الألوان (CFA)",
               "يتحقق من أن نمط استكمال الألوان في الكاميرا يقع على الشبكة نفسها في كل الصورة؛ "
               "وأي مجموعة مربعات على الشبكة المعاكسة مصدرها صورة أخرى.",
               "يصلح فقط للصور غير المضغوطة أو العالية الجودة جدًا التي لم يُغيَّر حجمها؛ والمنطقة "
               "الملصوقة بالطور نفسه لا تُرى، فيفوته بحكم تصميمه نحو نصف حالات اللصق.")},
    "image.noise": {
        "en": ("Estimates the noise in each block, compares it with what brightness and "
               "texture predict, and flags a compact cluster of abnormally smooth blocks "
               "(in-painting, heavy retouching).",
               "A weak vote: smooth regions also occur naturally."),
        "ar": ("اتساق ضجيج المستشعر",
               "يقدّر الضجيج في كل مربع ويقارنه بما تتوقعه الإضاءة والتفاصيل، ويشير إلى أي "
               "مجموعة متراصّة من المربعات الملساء على نحو غير طبيعي (ترميم أو تنعيم شديد).",
               "قرينة ضعيفة: فالمناطق الملساء تظهر طبيعيًا أيضًا.")},
    "image.spectral": {
        "en": ("Looks for sharp periodic peaks in the spectrum of the noise residual, the "
               "imprint of a generator's upsampling layers.",
               "Disabled for lossy files (JPEG blocking occupies the same frequencies), needs "
               "a file that was not resized, and has not been validated on real generator "
               "output."),
        "ar": ("بصمة المولِّد في الطيف",
               "يبحث عن قمم دورية حادّة في طيف بقايا الضجيج، وهي بصمة طبقات التكبير في "
               "المولِّدات.",
               "يُعطَّل في الملفات المضغوطة بفقد (لأن مربعات JPEG تشغل الترددات نفسها)، ويحتاج إلى "
               "ملف لم يُغيَّر حجمه، ولم يُتحقَّق منه على مخرجات مولِّدات حقيقية.")},
    "image.ela": {
        "en": ("Re-saves the image as JPEG and maps where the re-save error departs from what "
               "the local texture predicts.",
               "Low weight: kept mainly as a familiar visual, and meaningful for JPEG files "
               "only."),
        "ar": ("تحليل مستوى الخطأ (ELA)",
               "يعيد حفظ الصورة بصيغة JPEG ويرسم المواضع التي يختلف فيها خطأ إعادة الحفظ عمّا "
               "تتوقعه التفاصيل المحلية.",
               "وزنه منخفض: أُبقي عليه أساسًا لأنه عرض مألوف، ولا معنى له إلا في ملفات JPEG.")},
    "image.thumbnail": {
        "en": ("Compares the thumbnail embedded in the EXIF data with the image itself; a "
               "difference means the image changed after the thumbnail was written.",
               "Applies only when the file still carries an embedded thumbnail."),
        "ar": ("مطابقة الصورة المصغّرة في EXIF",
               "يقارن الصورة المصغّرة المضمَّنة في بيانات EXIF بالصورة نفسها؛ والاختلاف يعني أن "
               "الصورة تغيّرت بعد كتابة المصغّرة.",
               "لا ينطبق إلا إذا كان الملف ما زال يحمل صورة مصغّرة مضمَّنة.")},
    "image.generator_metadata": {
        "en": ("Reads what the file says about itself: generator text chunks (Stable Diffusion "
               "WebUI, ComfyUI, InvokeAI, NovelAI, validated structurally), the IPTC digital "
               "source type and C2PA AI assertions.",
               "Metadata is trivially stripped, so its absence proves nothing."),
        "ar": ("بيانات المولِّد الوصفية",
               "يقرأ ما يصرّح به الملف عن نفسه: البيانات النصية للمولِّدات (Stable Diffusion "
               "WebUI وComfyUI وInvokeAI وNovelAI، مع التحقق من بنيتها)، ونوع المصدر الرقمي في "
               "IPTC، وتصريحات C2PA المتعلقة بالذكاء الاصطناعي.",
               "البيانات الوصفية تُحذف بسهولة، فغيابها لا يثبت شيئًا.")},
    "image.provenance": {
        "en": ("Reads editing-software tags, the XMP edit history, C2PA presence and "
               "mismatches between capture and modification dates.",
               "Describes the file's history, not its pixels; metadata is easily stripped."),
        "ar": ("البيانات الوصفية والمصدر",
               "يقرأ وسوم برامج التحرير، وسجلّ التعديلات في XMP، ووجود C2PA، واختلاف تاريخ "
               "التعديل عن تاريخ الالتقاط.",
               "يصف تاريخ الملف لا محتوى الصورة، والبيانات الوصفية تُحذف بسهولة.")},
    "image.file_prior": {
        "en": ("A weak contextual prior from the shape of the file: no camera EXIF, a "
               "resolution common among generators or dimensions that are multiples of 64, a "
               "PNG without camera data.",
               "A hint only, with a fixed low confidence (0.2); it says nothing about the "
               "pixels."),
        "ar": ("قرائن شكل الملف",
               "قرينة سياقية ضعيفة من شكل الملف: غياب بيانات الكاميرا، أو أبعاد شائعة لدى "
               "المولِّدات أو من مضاعفات 64، أو ملف PNG بلا بيانات كاميرا.",
               "مجرّد تلميح بثقة منخفضة ثابتة (0.2)، ولا يقول شيئًا عن محتوى الصورة.")},
    "audio.enf": {
        "en": ("Isolates the mains hum (50/60 Hz or its second harmonic) and follows its "
               "phase; a cut shows up as a phase step, a frequency step or a dropout.",
               "Not applicable when the recording carries no measurable hum (below 10 dB hum "
               "SNR)."),
        "ar": ("استمرارية طنين الشبكة الكهربائية (ENF)",
               "يعزل طنين الشبكة الكهربائية (50 أو 60 هرتز أو توافقيته الثانية) ويتتبّع طوره؛ "
               "والقطع يظهر قفزةً في الطور أو في التردد أو انقطاعًا في الطنين.",
               "لا ينطبق إذا لم يحمل التسجيل طنينًا قابلًا للقياس (نسبة الطنين إلى الضجيج أقل من "
               "10 ديسيبل).")},
    "audio.splice": {
        "en": ("Compares the noise floor and the average spectrum of the pauses between "
               "neighbouring windows, and looks for the click a hard cut leaves.",
               "Needs pauses to examine, and has been measured on synthetic recordings only."),
        "ar": ("اتساق الخلفية الصوتية",
               "يقارن مستوى الضجيج الخلفي ومتوسط طيف فترات الصمت بين النوافذ المتجاورة، ويبحث "
               "عن النقرة التي يتركها القطع الحادّ.",
               "يحتاج إلى فترات صمت يفحصها، ولم يُقَس أداؤه إلا على تسجيلات اصطناعية.")},
    "audio.silence": {
        "en": ("Finds runs of identical samples lasting 20 ms or more inside a recording that "
               "otherwise has a noise floor.",
               "Carries no weight when the whole file is noise-gated or synthesised, because "
               "digital silence is then everywhere."),
        "ar": ("صمت رقمي مُقحَم",
               "يبحث عن سلاسل من العيّنات المتطابقة مدتها 20 ميلي ثانية أو أكثر داخل تسجيل له "
               "ضجيج خلفي في بقية أجزائه.",
               "لا وزن له إذا كان الملف كله معالَجًا ببوابة ضجيج أو مولَّدًا، لأن الصمت الرقمي "
               "يكون حينها في كل مكان.")},
    "audio.bandwidth": {
        "en": ("Looks for a brick-wall cutoff far below the Nyquist frequency, which means the "
               "audio was produced at a lower sample rate and then upsampled.",
               "Any upsampled or transcoded source looks the same, so it is not specific to "
               "synthetic speech."),
        "ar": ("عرض النطاق الفعلي",
               "يبحث عن انقطاع حادّ في الطيف دون تردد نايكويست بكثير، ما يعني أن الصوت أُنتج "
               "بمعدل عيّنات أدنى ثم رُفع.",
               "كل مصدر رُفع معدل عيّناته أو أُعيد ترميزه يبدو كذلك، فهو ليس خاصًّا بالكلام "
               "الاصطناعي.")},
    "audio.bandwidth_shift": {
        "en": ("Tracks the cutoff frequency over time; a passage with a different cutoff "
               "points to material from another source spliced in.",
               "Lossy codecs vary their high-frequency band on their own, so the test is far "
               "less strict on lossy files."),
        "ar": ("ثبات عرض النطاق عبر الزمن",
               "يتتبّع تردد القطع عبر الزمن؛ والمقطع الذي يختلف تردد قطعه يشير إلى مادة من "
               "مصدر آخر أُدرجت في التسجيل.",
               "الترميز بفقد يغيّر نطاق الترددات العالية من تلقاء نفسه، فيكون الاختبار أقل "
               "صرامة بكثير في الملفات المضغوطة بفقد.")},
    "audio.synthetic": {
        "en": ("Combines cues typical of synthetic speech: unnaturally clean pauses, a flat "
               "noise floor and persistent high-frequency lines.",
               "Low weight by design: the same cues fire on noise-gated and phone audio."),
        "ar": ("قرائن الكلام الاصطناعي",
               "يجمع قرائن مألوفة في الكلام الاصطناعي: فترات صمت نظيفة على نحو غير طبيعي، "
               "وضجيج خلفي ثابت، وخطوط مستمرة في الترددات العالية.",
               "وزنه منخفض عمدًا: فالقرائن نفسها تظهر في الصوت المعالَج ببوابة ضجيج وفي "
               "تسجيلات الهاتف.")},
    "audio.metadata": {
        "en": ("Reads encoder, editor and generator tags and IPTC / C2PA markers from the "
               "audio file.",
               "Metadata is easy to strip, so finding nothing is not evidence of "
               "authenticity."),
        "ar": ("البيانات الوصفية للصوت",
               "يقرأ من الملف الصوتي وسوم المرمِّز وبرامج التحرير والمولِّدات وعلامات IPTC "
               "وC2PA.",
               "البيانات الوصفية سهلة الحذف، فعدم العثور على شيء ليس دليلًا على الأصالة.")},
    "video.jump": {
        "en": ("Looks for a single-frame spike in the difference between consecutive frames "
               "while the colour histogram says the scene has not changed: the trace of "
               "removed frames.",
               "A deliberate jump cut looks exactly the same."),
        "ar": ("قفزات زمنية (إطارات محذوفة)",
               "يبحث عن قفزة في إطار واحد في الفرق بين الإطارات المتتالية بينما يدلّ توزيع "
               "الألوان على أن المشهد لم يتغيّر: وهذا أثر حذف إطارات.",
               "القطع المتعمَّد في المونتاج يبدو مطابقًا تمامًا.")},
    "video.repeat": {
        "en": ("Compares compact signatures of all frames with one another and reports runs of "
               "8 or more frames that reappear later while the content is moving.",
               "Not applicable to footage that is nearly static or too short."),
        "ar": ("لقطات معادة",
               "يقارن بصمات مختصرة لكل الإطارات بعضها ببعض، ويبلّغ عن سلاسل من 8 إطارات أو "
               "أكثر تعود للظهور لاحقًا والمحتوى متحرك.",
               "لا ينطبق على اللقطات شبه الثابتة ولا على المقاطع القصيرة جدًا.")},
    "video.freeze": {
        "en": ("Finds runs of 3 or more identical frames inside moving footage; regular single "
               "duplicates are reported as frame-rate conversion instead.",
               "Skipped when the video had to be sub-sampled; static footage repeats frames "
               "naturally."),
        "ar": ("إطارات مجمَّدة أو مكرَّرة",
               "يبحث عن سلاسل من 3 إطارات متطابقة أو أكثر داخل لقطات متحركة؛ أما التكرار "
               "المفرد المنتظم فيُبلَّغ عنه بوصفه تحويلًا لمعدل الإطارات.",
               "يُتخطّى إذا اضطُرّ التحليل إلى أخذ عيّنة من الإطارات؛ واللقطات الثابتة تكرّر "
               "إطاراتها طبيعيًا.")},
    "video.source": {
        "en": ("Compares the median noise level of each shot with the rest of the video, to "
               "spot footage from another source.",
               "Weak: video codecs flatten the noise differences between sources."),
        "ar": ("اتساق المصدر (مستوى الضجيج)",
               "يقارن وسيط مستوى الضجيج في كل لقطة ببقية الفيديو لرصد مادة من مصدر آخر.",
               "ضعيف: فمرمِّزات الفيديو تطمس فروق الضجيج بين المصادر.")},
    "video.double_compression": {
        "en": ("Reads the frame sizes from the MP4 sample table and looks for the periodic "
               "pattern left by the key-frame spacing (GOP) of an earlier encoding.",
               "Unreliable when the stream uses B-frames, and needs an MP4/MOV sample table."),
        "ar": ("ضغط مزدوج",
               "يقرأ أحجام الإطارات من جدول العيّنات في MP4 ويبحث عن النمط الدوري الذي خلّفه "
               "تباعد الإطارات المفتاحية (GOP) في ترميز سابق.",
               "غير موثوق إذا كان المسار يستخدم إطارات B، ويحتاج إلى جدول عيّنات MP4/MOV.")},
    "video.frame_copymove": {
        "en": ("Runs the image copy-move detector on key frames and reports a clone only if it "
               "persists across them.",
               "Same limits as on images: needs texture and cannot tell a digital clone from "
               "identical printed objects."),
        "ar": ("النسخ واللصق في الإطارات المفتاحية",
               "يشغّل كاشف النسخ واللصق على الإطارات المفتاحية، ولا يبلّغ عن نسخة إلا إذا "
               "استمرت عبرها.",
               "له حدود كاشف الصور نفسها: يحتاج إلى تفاصيل، ولا يميّز النسخ الرقمي من أجسام "
               "مطبوعة متطابقة.")},
    "video.provenance": {
        "en": ("Reads container tags, edit lists and transcoder markers.",
               "A transcoder tag is normal for anything downloaded from the web, and tags are "
               "easily stripped."),
        "ar": ("الحاوية والمصدر",
               "يقرأ وسوم الحاوية وقوائم التحرير وعلامات برامج إعادة الترميز.",
               "وسم إعادة الترميز أمر عادي في كل ما يُنزَّل من الإنترنت، والوسوم تُحذف بسهولة.")},
    "video.generator_metadata": {
        "en": ("Looks in the container for AI-generator markers and Content Credentials "
               "(C2PA) AI assertions.",
               "Metadata is easily stripped, and platforms remove it on upload."),
        "ar": ("بيانات المولِّد الوصفية",
               "يبحث في الحاوية عن علامات مولِّدات الذكاء الاصطناعي وعن تصريحات بيانات "
               "الاعتماد (C2PA) المتعلقة به.",
               "البيانات الوصفية تُحذف بسهولة، والمنصّات تزيلها عند الرفع.")},
    "video.decode": {
        "en": ("The frame-decoding step that every content detector for video depends on.",
               "If it fails, no detector could look at the frames."),
        "ar": ("فكّ ترميز الإطارات",
               "خطوة فكّ ترميز الإطارات التي تعتمد عليها كل كواشف محتوى الفيديو.",
               "إذا فشلت فلن يتمكن أي كاشف من فحص الإطارات.")},
    "video.audio_track": {
        "en": ("Extracts the sound track and hands it to the audio engine.",
               "If it fails, the sound track was not examined."),
        "ar": ("المسار الصوتي",
               "يستخرج المسار الصوتي ويسلّمه إلى محرّك الصوت.",
               "إذا فشل فمعناه أن المسار الصوتي لم يُفحص.")},
}

VERDICT_KEYS = {
    "No significant traces found": "v0",
    "Inconclusive (weak anomalies)": "v1",
    "Inconclusive (content could not be analysed)": "v1_unreadable",
    "Inconclusive (part of the analysis failed)": "v1_partial",
    "Suspicious": "v2",
    "Likely manipulated": "v3_manip",
    "Likely AI-generated": "v3_ai",
    "AI-generated (provenance evidence)": "v4_ai",
    "Edited (provenance evidence)": "v4_edit",
}
_LEVEL_FALLBACK = {0: "v0", 1: "v1", 2: "v2", 3: "v3_manip", 4: "v4_edit"}

# --------------------------------------------------------------------------- #
# Stylesheet (shared with the upload page) and the one static script
# --------------------------------------------------------------------------- #
_LAYER_RULES = ",".join(f".viewer:has(.lt{i}:checked) .l{i}" for i in range(MAX_LAYERS))

CSS = """
:root{color-scheme:light dark;--bg:#f3f2ee;--surface:#ffffff;--sunk:#efede7;--ink:#1a1917;
--ink2:#57544d;--line:#d9d6cc;--accent:#1d4f91;--l0:#3f4c5e;--l1:#6b5a10;--l2:#96490a;
--l3:#a4271c;--l4:#8a1c2e;--t0:#eef1f5;--t1:#f6f1dc;--t2:#fbeedd;--t3:#fbe9e6;--t4:#f9e6ea;
--mark:#c2302b;--plot:#256abf}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--surface:#1b1b19;--sunk:#272623;
--ink:#edebe5;--ink2:#aaa69c;--line:#3a3833;--accent:#9dbdf5;--l0:#b3c0d2;--l1:#dcc96f;
--l2:#f2ab62;--l3:#ff8b7d;--l4:#ff8fa3;--t0:#20252c;--t1:#2a2613;--t2:#2f2213;--t3:#331a17;
--t4:#33181e;--mark:#ff7b6e;--plot:#6da7ec}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-padding-top:12px}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
:lang(ar){font-family:"SF Arabic","Geeza Pro","Segoe UI","Noto Naskh Arabic","Noto Sans Arabic",
Tahoma,system-ui,sans-serif;line-height:1.8}
[dir=ltr]:lang(en){font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,
sans-serif;line-height:1.55}
code,pre,.mono{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
font-size:.92em}
a{color:var(--accent)}
:focus-visible{outline:3px solid var(--accent);outline-offset:2px;border-radius:3px}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px}
.skip{position:absolute;inset-inline-start:8px;top:-80px;background:var(--surface);
padding:8px 14px;z-index:9}
.skip:focus{top:8px}
.mast{display:flex;gap:14px;align-items:baseline;justify-content:space-between;flex-wrap:wrap;
padding:22px 0 12px;border-bottom:2px solid var(--ink)}
.brand{font-weight:700;letter-spacing:.02em;font-size:17px}
.brand span{font-weight:400;color:var(--ink2)}
h1{font-size:15px;font-weight:600;margin:0;color:var(--ink2)}
h2{font-size:20px;line-height:1.3;margin:40px 0 4px}
h3{font-size:16px;margin:0}h4,h5,h6{font-size:15px;margin:0}
p{margin:6px 0}
.lead{color:var(--ink2);margin:0 0 12px;max-width:78ch}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px;
margin:12px 0}
.btn{font:inherit;font-size:13.5px;color:var(--ink);background:var(--surface);
border:1px solid var(--ink2);border-radius:6px;padding:5px 12px;cursor:pointer;
text-decoration:none;display:inline-block}
.btn:hover{background:var(--sunk)}
.btn[aria-pressed=true]{background:var(--ink);color:var(--surface)}
.js-only{display:none}.js .js-only{display:revert}
.muted{color:var(--ink2)}.small{font-size:13px}
nav.toc{font-size:13.5px;padding:10px 0 0}
nav.toc ul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:4px 18px}
/* file facts */
.facts{display:grid;grid-template-columns:max-content minmax(0,1fr) max-content minmax(0,1fr);
gap:4px 18px;margin:0}
.facts dt{color:var(--ink2);font-size:13px;padding-top:2px}
.facts dd{margin:0;overflow-wrap:anywhere}.facts dd.wide{grid-column:2/-1}
@media (max-width:820px){.facts{grid-template-columns:max-content minmax(0,1fr)}}
.facts .name{font-weight:650;font-size:17px}
.hash{user-select:all;-webkit-user-select:all}
/* verdict */
.verdict{border-inline-start:6px solid var(--lv);background:var(--surface)}
.lv0{--lv:var(--l0);--tint:var(--t0)}.lv1{--lv:var(--l1);--tint:var(--t1)}
.lv2{--lv:var(--l2);--tint:var(--t2)}.lv3{--lv:var(--l3);--tint:var(--t3)}
.lv4{--lv:var(--l4);--tint:var(--t4)}
.vhead{display:flex;gap:16px;align-items:flex-start}
.vicon{flex:none;width:46px;height:46px;color:var(--lv)}
.vicon svg{width:100%;height:100%;display:block}
.over{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink2);
font-weight:600}
:lang(ar) .over{letter-spacing:0;font-size:13px}
.vtext{font-size:28px;line-height:1.2;font-weight:700;color:var(--lv);margin:2px 0 0}
.vsub{font-size:13px;color:var(--ink2)}
.scale{list-style:none;display:grid;grid-template-columns:repeat(5,1fr);gap:4px;padding:0;
margin:16px 0 0;font-size:12px;color:var(--ink2)}
.scale li{border-top:5px solid var(--line);padding-top:5px;line-height:1.3}
.scale li[aria-current]{border-top-color:var(--lv);color:var(--ink);font-weight:650}
.scale b{font-weight:inherit;margin-inline-end:4px}
.mean{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;
margin-top:16px}
.mean>div{background:var(--tint);border-radius:6px;padding:12px 14px}
.mean>div.not{background:var(--sunk)}
.mean h3{font-size:13px;margin:0 0 4px}
.mean p{margin:0}
.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px 26px;
margin-top:6px}
.sc .lab{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.sc b{font-size:20px;font-variant-numeric:tabular-nums}
.bar{position:relative;height:10px;border-radius:5px;background:var(--sunk);margin-top:4px;
border:1px solid var(--line)}
.bar i{position:absolute;inset-block:0;inset-inline-start:0;border-radius:4px;
background:var(--ink2);min-width:0}
.bar i.hot{background:var(--lv,var(--mark))}
.bar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.55}
.sub{margin:22px 0 6px}
.why{margin:6px 0 0;padding-inline-start:20px}.why li{margin:8px 0}
.why .src{font-weight:650}
.say{display:block;unicode-bidi:isolate;text-align:start}
[dir=rtl] .say[dir=ltr]{text-align:left;border-inline-start:3px solid var(--line);
padding-inline-start:10px;margin-top:2px}
/* viewer */
.viewer{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px;align-items:start}
.vside{display:grid;gap:14px;text-align:start}
.stage-scroll{overflow:auto;border-radius:6px;background:var(--sunk);line-height:0}
.zoomed .stage-scroll{max-height:82vh}
.stage{position:relative;width:100%;margin:0 auto;cursor:zoom-in;--op:.8;--split:0%}
.zoomed .stage{width:250%;max-width:none!important;cursor:zoom-out}
.stage img{display:block;width:100%;height:auto}
.stage .px{image-rendering:pixelated}
.ov{position:absolute;inset:0;clip-path:inset(0 0 0 var(--split))}
.layer,.ov svg{position:absolute;inset:0;width:100%;height:100%}
.layer{display:none;opacity:var(--op)}
.layer.dflt{display:block}
.divider{position:absolute;top:0;bottom:0;left:var(--split);width:2px;margin-left:-1px;
background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.6);display:none}
.wiping .divider{display:block}
@supports selector(:has(a)){.layer.dflt{display:none}
""" + _LAYER_RULES + """{display:block}
.viewer:has(.ltb:not(:checked)) .boxes{display:none}}
.box rect.halo{fill:none;stroke:#fff;stroke-width:5px;vector-effect:non-scaling-stroke}
.box rect.edge{fill:none;stroke:var(--c);stroke-width:2.5px;vector-effect:non-scaling-stroke}
.box.hl rect.edge{stroke-width:5px}
.box .tag{fill:#fff;stroke:var(--c);stroke-width:3px;vector-effect:non-scaling-stroke}
.box text{fill:#111;font-weight:700;font-family:system-ui,sans-serif;text-anchor:middle}
fieldset{border:0;margin:0;padding:0;min-width:0}
legend{font-weight:650;font-size:13px;padding:0;margin-bottom:6px}
.opt{display:flex;gap:8px;align-items:flex-start;margin:5px 0;font-size:13.5px;line-height:1.35}
.opt input{margin:2px 0 0;flex:none;width:16px;height:16px;accent-color:var(--ink)}
.sw{flex:none;width:14px;height:14px;border-radius:3px;margin-top:2px;background:var(--c);
border:1px solid rgba(0,0,0,.35)}
.sw.ramp{width:34px;background:linear-gradient(90deg,#16103a,#6a176e,#bc3754,#f37819,#f6d746)}
.ctl label{display:block;font-weight:650;font-size:13px}
.ctl input[type=range]{width:100%;accent-color:var(--ink);margin:4px 0 0}
.num{flex:none;display:inline-flex;align-items:center;justify-content:center;min-width:20px;
height:20px;padding:0 4px;border-radius:10px;background:#fff;color:#111;
border:2.5px solid var(--c,var(--mark));font-size:12px;font-weight:700;line-height:1}
.rlist{list-style:none;margin:0;padding:0;font-size:13.5px}
.rlist li{display:flex;gap:8px;align-items:flex-start;padding:6px;border-radius:6px;margin:0 -6px}
.rlist li.hl{background:var(--sunk)}
/* timeline */
.tl{position:relative}
.spec{display:block;width:100%;height:150px;border-radius:4px}
.specbox{position:relative;line-height:0;margin-bottom:8px}
.specbox span{position:absolute;left:4px;font-size:11px;line-height:1.2;color:#fff;
background:rgba(0,0,0,.6);padding:1px 5px;border-radius:3px}
.specbox .hi{top:4px}.specbox .lo{bottom:4px}
.lane{margin-top:8px}
.lane .nm{font-size:12.5px;color:var(--ink2);line-height:1.3;display:block;text-align:start}
.track{position:relative;height:16px;background:var(--sunk);border:1px solid var(--line);
border-radius:3px;margin-top:2px}
.lane.all .track{height:22px}
.seg{position:absolute;top:-1px;bottom:-1px;min-width:4px;background:var(--mark);border-radius:2px}
.seg b{position:absolute;bottom:100%;left:50%;transform:translate(-50%,-2px);font-size:11px;
line-height:1;color:var(--ink);font-weight:700}
.lane.all{margin-top:20px}
.axis{position:relative;height:22px;margin-top:4px;font-size:11.5px;color:var(--ink2);
font-variant-numeric:tabular-nums}
.axis span{position:absolute;top:6px;transform:translateX(-50%);white-space:nowrap}
.axis span:first-child{transform:none}.axis span.end{transform:translateX(-100%)}
.axis span::before{content:"";position:absolute;left:50%;top:-6px;height:5px;width:1px;
background:var(--ink2)}
.axis span:first-child::before{left:0}.axis span.end::before{left:100%}
@media (max-width:560px){.axis span:nth-child(even):not(.end){display:none}}
.nw{white-space:nowrap}
/* tables */
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:start;vertical-align:top;padding:7px 12px 7px 0;border-bottom:1px solid var(--line)}
[dir=rtl] th,[dir=rtl] td{padding:7px 0 7px 12px}
th{font-size:12.5px;color:var(--ink2);font-weight:600;white-space:nowrap}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.resp td:last-child{width:46%}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
table.kv{width:auto}table.kv th{white-space:normal;font-size:13.5px;font-weight:400}
.st{display:inline-flex;align-items:center;gap:5px;font-size:12.5px;font-weight:650;
white-space:nowrap}
.st::before{content:"";width:9px;height:9px;border-radius:50%;border:2px solid currentColor}
.st.fired{color:var(--l3)}.st.fired::before{background:currentColor}
.st.quiet{color:var(--ink2)}
.st.na{color:var(--ink2)}.st.na::before{border-style:dotted}
.st.failed{color:var(--l2)}.st.failed::before{border-radius:0;background:currentColor;
transform:rotate(45deg)}
/* detector groups and cards */
details.grp{margin:14px 0}
details.grp>summary{cursor:pointer;padding:8px 0;border-bottom:1px solid var(--line);
display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;list-style:none}
details.grp>summary::-webkit-details-marker{display:none}
details.grp>summary::before{content:"+";font-weight:700;width:14px;flex:none}
details.grp[open]>summary::before{content:"\\2212"}
details.grp>summary>*{margin:0}
.det{margin:12px 0}
.dhead{display:flex;gap:8px 12px;align-items:baseline;flex-wrap:wrap}
.dhead .en{font-size:12.5px;color:var(--ink2)}
.chip{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:1px 9px;
color:var(--ink2);white-space:nowrap}
.chip.dec{border-color:var(--l4);color:var(--l4);font-weight:650}
.about{margin:10px 0 0;padding:0;display:grid;grid-template-columns:max-content minmax(0,1fr);
gap:2px 12px;font-size:13.5px;color:var(--ink2)}
.about dt{font-weight:650}.about dd{margin:0}
.gauges{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px 22px;
margin:12px 0 0;font-size:13px}
.gauges .lab{display:flex;justify-content:space-between;gap:8px;color:var(--ink2)}
.gauges .lab b{color:var(--ink);font-variant-numeric:tabular-nums}
.gauges .bar{height:8px}
.finding{margin-top:12px}
.finding .k{font-size:12.5px;font-weight:650;color:var(--ink2)}
.errbox{border:1px solid var(--l2);border-radius:6px;padding:8px 12px;margin-top:10px}
.figs{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.figs figure{margin:0;max-width:340px}
.figs img{max-width:100%;height:auto;border-radius:4px;display:block;border:1px solid var(--line)}
figcaption{font-size:12.5px;color:var(--ink2);margin-top:3px}
.js .zoomable{cursor:zoom-in}
details.raw{margin-top:12px;font-size:13px}
details.raw summary{cursor:pointer;color:var(--ink2)}
pre{background:var(--sunk);padding:10px 12px;border-radius:6px;overflow:auto;max-height:320px;
margin:6px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;text-align:left}
/* plots */
.plot{margin:14px 0 0}
.plot figcaption{margin:0 0 3px;display:flex;gap:10px;justify-content:space-between;
flex-wrap:wrap}
.plot figcaption b{color:var(--ink);font-weight:650}
.pbox{position:relative;padding-left:52px}
.pbox .y{position:absolute;left:0;width:46px;text-align:right;font-size:11px;color:var(--ink2);
font-variant-numeric:tabular-nums;line-height:1}
.pbox .y.hi{top:0}.pbox .y.lo{top:86px}
.psvg{display:block;width:100%;height:96px;background:var(--sunk);border:1px solid var(--line);
border-radius:3px;overflow:visible;touch-action:pan-y}
.psvg path,.psvg line{vector-effect:non-scaling-stroke;fill:none}
.psvg .ln{stroke:var(--plot);stroke-width:1.6px;stroke-linejoin:round}
.psvg .mk{stroke:var(--mark);stroke-width:1.5px;stroke-dasharray:4 3}
.psvg .cur{stroke:var(--ink);stroke-width:1px;display:none}
.px-axis{display:flex;justify-content:space-between;font-size:11px;color:var(--ink2);
font-variant-numeric:tabular-nums;margin-top:2px}
.tip{position:absolute;top:-6px;transform:translate(-50%,-100%);background:var(--ink);
color:var(--surface);font-size:12px;line-height:1.3;padding:3px 8px;border-radius:4px;
white-space:nowrap;pointer-events:none;font-variant-numeric:tabular-nums;z-index:2}
.mkcap{font-size:12px;color:var(--ink2);margin-top:2px;text-align:left}
.mkcap i{display:inline-block;width:14px;border-top:2px dashed var(--mark);vertical-align:middle;
margin-right:5px}
/* how the verdict was reached */
.fhead{display:flex;gap:10px;align-items:baseline;justify-content:space-between;flex-wrap:wrap}
.fscroll{overflow-x:auto;line-height:0}
.fsvg{display:block;width:100%;min-width:760px;height:auto;line-height:normal}
.fsvg text{fill:var(--ink);font-size:12.5px}
.fsvg .hd{fill:var(--ink2);font-size:11px;font-weight:600;letter-spacing:.08em;
text-transform:uppercase}
:lang(ar) .fsvg .hd{letter-spacing:0;text-transform:none;font-size:12.5px}
.fsvg .nm{font-weight:650}.fsvg .sm{font-size:11px;fill:var(--ink2)}
.fsvg .n{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
font-size:11.5px;font-variant-numeric:tabular-nums}
.fsvg .big{font-size:17px;font-weight:700}
.fsvg .box{fill:var(--surface);stroke:var(--line);stroke-width:1.2px}
.fsvg .fired .box,.fsvg .cat.hot .box{stroke:var(--l3);stroke-width:2px}
.fsvg .failed .box{stroke:var(--l2);stroke-dasharray:5 3}
.fsvg .na .box,.fsvg .cat.prov .box{stroke-dasharray:3 3}
.fsvg .trk{fill:var(--sunk);stroke:var(--line);stroke-width:1px}
.fsvg .fill{fill:var(--ink2)}.fsvg .fired .fill,.fsvg .hot .fill{fill:var(--mark)}
.fsvg .tick{stroke:var(--ink);stroke-width:1.5px;opacity:.6}
.fsvg .s-fired{fill:var(--l3);font-weight:650}.fsvg .s-failed{fill:var(--l2);font-weight:650}
.fsvg .s-quiet,.fsvg .s-na{fill:var(--ink2)}
.fsvg .e{fill:none;stroke:var(--ink2);stroke-width:1px;opacity:.4}
.fsvg .e.in{opacity:.25}
.fsvg .e.fired,.fsvg .e.hot{stroke:var(--mark);opacity:.85}
.fsvg .e.na{stroke-dasharray:2 4;opacity:.3}
.fsvg .e.failed{stroke:var(--l2);stroke-dasharray:6 4;opacity:.75}
.fsvg .e.prov{stroke-dasharray:6 4}
.fsvg .pl{fill:none;stroke:var(--mark);stroke-linecap:round;stroke-dasharray:.04 .96;opacity:0}
.fsvg .elab{fill:var(--l3);font-size:11px;font-weight:700;paint-order:stroke;
stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.fsvg .vd .box{stroke:var(--lv);stroke-width:2.5px}
.fsvg .vd .big{fill:var(--lv)}
.fsvg .stp{fill:var(--sunk);stroke:var(--line);stroke-width:1px}
.fsvg .stp.on{fill:var(--lv);stroke:var(--lv)}
.fsvg a:hover .box{fill:var(--sunk)}
.fsvg .nd{transition:opacity .15s}
.fsvg:has(.nd:hover) .nd:not(:hover){opacity:.25}
.fkey{list-style:none;display:flex;flex-wrap:wrap;gap:4px 20px;margin:10px 0 0;padding:0;
font-size:12.5px;color:var(--ink2)}
.fkey i{display:inline-block;width:26px;border-top:2px solid var(--ink2);vertical-align:middle;
margin-inline-end:6px}
.fkey .k-w{border-top:5px solid var(--mark)}.fkey .k-quiet{border-top-width:1px}
.fkey .k-na{border-top-style:dotted}.fkey .k-failed{border-top:2px dashed var(--l2)}
ol.fsteps{margin:14px 0 0;padding-inline-start:22px}ol.fsteps li{margin:5px 0}
@media (prefers-reduced-motion:no-preference){
.flow.play .an{animation:fin .5s var(--d,0s) both}
.flow.play .pl{animation:run 1.1s var(--d,0s) 2 linear}
.flow.play ol.fsteps li{animation:dim .5s var(--d,0s) both}
@keyframes fin{from{opacity:0}}
@keyframes dim{from{opacity:.25}}
@keyframes run{from{opacity:1;stroke-dashoffset:1}to{opacity:1;stroke-dashoffset:0}}}
/* child report, footer, lightbox */
.child{border-inline-start:3px solid var(--line);padding-inline-start:18px;margin-top:8px}
.child .vtext{font-size:21px}.child .vicon{width:36px;height:36px}
footer.foot{margin:44px 0 0;padding:16px 0 56px;border-top:2px solid var(--ink);
color:var(--ink2);font-size:13.5px}
footer.foot h2{font-size:15px;margin:0 0 4px;color:var(--ink)}
footer.foot p{max-width:86ch}
dialog.lb{border:0;border-radius:8px;padding:14px;background:var(--surface);color:var(--ink);
max-width:96vw;max-height:94vh;overflow:auto}
dialog.lb::backdrop{background:rgba(0,0,0,.72)}
dialog.lb img{display:block;max-width:none;margin-top:10px}
dialog.lb p{margin:8px 0 0;font-size:13px;color:var(--ink2)}
@media (max-width:820px){.viewer{grid-template-columns:minmax(0,1fr)}
.vtext{font-size:23px}.scale{font-size:11px}
.resp thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
.resp tr{display:block;padding:8px 0;border-bottom:1px solid var(--line)}
.resp td,.resp th{display:block;border:0;padding:2px 0!important;width:auto!important}
.resp td[data-l]::before{content:attr(data-l) ": ";color:var(--ink2);font-size:12.5px}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}
.btn,.rlist li{transition:background-color .12s}}
@media (forced-colors:active){.bar i,.seg{background:Highlight}.bar u{background:CanvasText}
.layer{forced-color-adjust:none}}
@media print{:root{--bg:#ffffff;--surface:#ffffff;--sunk:#f3f2ee;--ink:#000000;--ink2:#3a3a3a;
--line:#b9b6ad;--accent:#000;--l0:#3f4c5e;--l1:#6b5a10;--l2:#96490a;--l3:#a4271c;--l4:#8a1c2e;
--t0:#eef1f5;--t1:#f6f1dc;--t2:#fbeedd;--t3:#fbe9e6;--t4:#f9e6ea;--mark:#c2302b;--plot:#256abf}
@page{margin:14mm}
body{font-size:11pt}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}
.wrap{max-width:none;padding:0}
.skip,.toc,.js-only,.noprint,details.raw,.vside .ctl,.vside fieldset,.tip{display:none!important}
.card{border-color:#999;break-inside:avoid;padding:10px 12px}
.card.verdict,section>.card{break-inside:auto}
.vhead,.mean>div,.scores,.why li{break-inside:avoid}h2,h3,h4,summary{break-after:avoid}
.viewer{grid-template-columns:minmax(0,1fr)}
.stage{max-width:150mm!important;width:100%!important}.stage-scroll{overflow:visible;
max-height:none!important;background:none}
.ov{clip-path:none}.divider{display:none!important}
.figs figure{max-width:80mm}tr,figure,.lane{break-inside:avoid}
pre{max-height:none}a{text-decoration:none;color:inherit}
.fscroll{overflow:visible}.fsvg{min-width:0}.flow .an,.flow .pl,ol.fsteps li{animation:none!important}
details.grp:not([open]){display:none}}
"""

JS = """(function(){'use strict';
var d=document,root=d.documentElement;root.classList.add('js');
function all(s,c){return Array.prototype.slice.call((c||d).querySelectorAll(s))}
function on(el,ev,fn){if(el)el.addEventListener(ev,fn)}
all('.viewer').forEach(function(v){
 var stage=v.querySelector('.stage'),sc=v.querySelector('.stage-scroll'),
  op=v.querySelector('.op'),sp=v.querySelector('.split'),zb=v.querySelector('.zoom-btn');
 on(op,'input',function(){stage.style.setProperty('--op',op.value/100)});
 on(sp,'input',function(){stage.style.setProperty('--split',sp.value+'%');
  v.classList.toggle('wiping',sp.value>0)});
 function zoom(z,fx,fy){v.classList.toggle('zoomed',z);
  if(zb)zb.setAttribute('aria-pressed',z?'true':'false');
  if(z){sc.scrollLeft=fx*sc.scrollWidth-sc.clientWidth/2;
   sc.scrollTop=fy*sc.scrollHeight-sc.clientHeight/2}}
 on(stage,'click',function(e){var r=stage.getBoundingClientRect();
  zoom(!v.classList.contains('zoomed'),(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height)});
 on(zb,'click',function(){zoom(!v.classList.contains('zoomed'),.5,.5)});
 all('[data-box]',v).forEach(function(li){var b=d.getElementById(li.getAttribute('data-box'));
  function hl(x){return function(){li.classList.toggle('hl',x);if(b)b.classList.toggle('hl',x)}}
  on(li,'mouseenter',hl(true));on(li,'mouseleave',hl(false))});
});
all('a[data-layer]').forEach(function(a){on(a,'click',function(){
 var c=d.getElementById(a.getAttribute('data-layer'));if(c)c.checked=true})});
var dlg=null;
function lightbox(img){
 if(!dlg){dlg=d.createElement('dialog');dlg.className='lb';
  var x=d.createElement('button');x.type='button';x.className='btn';
  x.textContent=root.getAttribute('data-close')||'Close';
  dlg.appendChild(x);dlg.appendChild(d.createElement('img'));dlg.appendChild(d.createElement('p'));
  d.body.appendChild(dlg);on(x,'click',function(){dlg.close()});
  on(dlg,'click',function(e){if(e.target===dlg)dlg.close()})}
 var i=dlg.querySelector('img');i.src=img.src;i.alt=img.alt;
 dlg.querySelector('p').textContent=img.getAttribute('data-cap')||img.alt;
 if(dlg.showModal)dlg.showModal();else dlg.setAttribute('open','')}
all('img.zoomable').forEach(function(img){img.tabIndex=0;img.setAttribute('role','button');
 img.setAttribute('aria-label',(root.getAttribute('data-enlarge')||'Enlarge')+': '+
  (img.getAttribute('data-cap')||img.alt));
 on(img,'click',function(){lightbox(img)});
 on(img,'keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();lightbox(img)}})});
function fmt(v){var a=Math.abs(v);return a>=1000?v.toFixed(0):a>=100?v.toFixed(1):a>=1?v.toFixed(2):
 v.toPrecision(3)}
all('svg.psvg').forEach(function(s){var pts=null,box=s.parentNode,tip=null,
 cur=s.querySelector('.cur'),g=function(k){return parseFloat(s.getAttribute('data-'+k))},
 x0=g('x0'),x1=g('x1'),y0=g('y0'),y1=g('y1'),xu=s.getAttribute('data-xu')||'';
 on(s,'pointermove',function(e){
  if(!pts){pts=[];all('.ln',s).forEach(function(p){(p.getAttribute('d').match(/-?[\\d.]+,-?[\\d.]+/g)
   ||[]).forEach(function(q){q=q.split(',');pts.push([+q[0],+q[1]])})});
   pts.sort(function(a,b){return a[0]-b[0]})}
  if(!pts.length)return;
  var r=s.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1000,lo=0,hi=pts.length-1;
  while(hi-lo>1){var m=(lo+hi)>>1;if(pts[m][0]<x)lo=m;else hi=m}
  var p=Math.abs(pts[lo][0]-x)<Math.abs(pts[hi][0]-x)?pts[lo]:pts[hi];
  if(!tip){tip=d.createElement('div');tip.className='tip';box.appendChild(tip)}
  tip.textContent=fmt(x0+p[0]/1000*(x1-x0))+' '+xu+'  \\u00b7  '+fmt(y1-p[1]/100*(y1-y0));
  tip.style.display='block';tip.style.left=(s.offsetLeft+p[0]/1000*r.width)+'px';
  cur.setAttribute('x1',p[0]);cur.setAttribute('x2',p[0]);cur.style.display='block'});
 on(s,'pointerleave',function(){if(tip)tip.style.display='none';cur.style.display='none'});
});
all('.flow').forEach(function(f){var b=f.parentNode.querySelector('.replay');
 function play(){f.classList.remove('play');void f.offsetWidth;f.classList.add('play')}
 on(b,'click',play);
 if(window.IntersectionObserver){var io=new IntersectionObserver(function(es){
  if(es[0].isIntersecting){io.disconnect();play()}},{threshold:.3});io.observe(f)}});
all('[data-print]').forEach(function(b){on(b,'click',function(){window.print()})});
})();"""

_JS_HASH = base64.b64encode(hashlib.sha256(JS.encode("utf-8")).digest()).decode("ascii")
CSP = ("default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
       f"script-src 'sha256-{_JS_HASH}'; base-uri 'none'; form-action 'none'")

_ICONS = {
    0: '<circle cx="24" cy="24" r="19" fill="none" stroke="currentColor" stroke-width="3.5"/>'
       '<path d="M14 24h20" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>',
    1: '<circle cx="24" cy="24" r="19" fill="none" stroke="currentColor" stroke-width="3.5"/>'
       '<path d="M18 19a6 6 0 1 1 8.5 5.5c-1.7.8-2.5 1.8-2.5 3.5" fill="none" '
       'stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>'
       '<circle cx="24" cy="34.5" r="2.2" fill="currentColor"/>',
    2: '<path d="M24 5 45 42H3Z" fill="none" stroke="currentColor" stroke-width="3.5" '
       'stroke-linejoin="round"/><path d="M24 19v11" stroke="currentColor" stroke-width="3.5" '
       'stroke-linecap="round"/><circle cx="24" cy="35.5" r="2.2" fill="currentColor"/>',
    3: '<path d="M24 5 45 42H3Z" fill="currentColor" stroke="currentColor" stroke-width="3.5" '
       'stroke-linejoin="round"/><path d="M24 19v11" stroke="var(--surface)" stroke-width="3.8" '
       'stroke-linecap="round"/><circle cx="24" cy="35.5" r="2.4" fill="var(--surface)"/>',
    4: '<path d="M11 4h18l9 9v31H11Z" fill="currentColor" stroke="currentColor" stroke-width="3" '
       'stroke-linejoin="round"/><path d="M17 22h15M17 29h15M17 36h9" stroke="var(--surface)" '
       'stroke-width="3.2" stroke-linecap="round"/>',
}


# --------------------------------------------------------------------------- #
# Escaping. Every value that can come from the analysed file passes through here.
# --------------------------------------------------------------------------- #
_BIDI = re.compile("[\u061c\u200e\u200f\u2028-\u202e\u2066-\u2069]")
_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff\ufffe\uffff]")


def _clean(x: Any) -> str:
    """Make hostile text inert *as text*: bidi controls (which can make ``gpj.exe`` read
    as ``exe.jpg``) and Unicode line separators become visible, control characters and
    lone surrogates (which would make the file unwritable as UTF-8) become U+FFFD."""
    s = x if isinstance(x, str) else str(x)
    s = _BIDI.sub(lambda m: "[U+%04X]" % ord(m.group()), s)
    return _CTRL.sub("\ufffd", s)


def _esc(x: Any) -> str:
    """Escape for HTML text and for double-quoted attribute values. The slashes of a
    ``scheme://`` are written as character references: the text reads and copies the same,
    but the file never contains a URL, so "no external reference" can be checked with a
    plain search."""
    return html.escape(_clean(x), quote=True).replace("://", ":&#47;&#47;")


def _ltr(x: Any, cls: str = "") -> str:
    """Numbers, file names and code keep left-to-right order inside right-to-left text."""
    c = f' class="{cls}"' if cls else ""
    return f'<bdi dir="ltr"{c}>{_esc(x)}</bdi>'


def _say(text: Any) -> str:
    """Engine-written English (summaries, labels): shown as is, isolated and left-to-right."""
    return f'<span class="say" dir="ltr" lang="en">{_esc(text)}</span>'


def _json_for_script(obj: Any) -> str:
    """JSON that is safe inside ``<script type="application/json">``: ``<``, ``>``, ``&``,
    line separators and bidi controls are written as ``\\uXXXX`` escapes, so ``</script>``
    and ``<!--`` cannot occur and the decoded values are unchanged."""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    s = _CTRL.sub("\ufffd", s)
    for ch in "<>&":
        s = s.replace(ch, "\\u%04x" % ord(ch))
    s = _BIDI.sub(lambda m: "\\u%04x" % ord(m.group()), s)   # values stay exact, source inert
    return s.replace("://", ":\\/\\/")


def _dom_id(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", x)


# --------------------------------------------------------------------------- #
# Rendering context
# --------------------------------------------------------------------------- #
class _Ctx:
    """Language, heading depth and id counter for one render."""

    def __init__(self, lang: str):
        self.lang = lang
        self.rtl = lang == "ar"
        self.depth = 0          # 0 = main report, 1 = child (embedded track)
        self.visual_bytes = VISUALS_TOTAL_BYTES     # what is left for detector visuals
        self._n = 0

    def t(self, key: str, **kw: Any) -> Any:
        v = STRINGS[self.lang].get(key, STRINGS["en"].get(key, key))
        return v.format(**kw) if kw and isinstance(v, str) else v

    def h(self, level: int) -> str:
        return f"h{min(6, level + self.depth)}"

    def uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def detector_name(self, e: Evidence) -> Tuple[str, str]:
        """(display name, English title to show next to it or '')."""
        if self.lang != "en" and e.detector in DETECTORS and self.lang in DETECTORS[e.detector]:
            return DETECTORS[e.detector][self.lang][0], e.title
        return e.title, ""

    def about(self, e: Evidence) -> Optional[Tuple[str, str]]:
        d = DETECTORS.get(e.detector)
        if not d:
            return None
        if self.lang != "en" and self.lang in d:
            return d[self.lang][1], d[self.lang][2]
        return d["en"][0], d["en"][1]


def _name_html(ctx: _Ctx, e: Evidence) -> str:
    name, en = ctx.detector_name(e)
    return _esc(name) if en else _ltr(name) if ctx.rtl else _esc(name)


# --------------------------------------------------------------------------- #
# Numbers and small helpers
# --------------------------------------------------------------------------- #
def _state(e: Evidence) -> str:
    """Outcome of one detector, with the same threshold the fusion uses."""
    if e.error:
        return "failed"
    if e.confidence <= 0:
        return "na"
    return "fired" if e.score >= 0.5 else "quiet"


def _weight(e: Evidence) -> float:
    """Share of the verdict this detector carries (docs/METHODS.md, "Fusion")."""
    return float(np.clip((e.score - 0.5) / 0.5, 0.0, 1.0)) * e.confidence


def _score(v: float) -> str:
    """Two decimals, but never print a score just under the threshold as 0.50."""
    s = f"{v:.2f}"
    return "0.49" if v < 0.5 and s == "0.50" else s


def _human_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{n} B"


def _num(v: float) -> str:
    """Compact number for axis labels."""
    if not np.isfinite(v):
        return "–"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def _clock(sec: float, total: float) -> str:
    """Time for humans: seconds below a minute, m:ss.s above."""
    if total < 60:
        return f"{sec:.2f} s"
    m, s = divmod(max(sec, 0.0), 60.0)
    return f"{int(m)}:{s:04.1f}"


def _sha256_of(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _verdict_key(r: Report) -> str:
    return VERDICT_KEYS.get(r.verdict) or _LEVEL_FALLBACK.get(int(r.verdict_level), "v1")


def _duration(r: Report) -> float:
    try:
        return float(r.info.get("duration") or r.info.get("duration_sec") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Images: one JPEG of the analysed picture, palette PNGs for the maps
# --------------------------------------------------------------------------- #
def _as_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(np.nan_to_num(a.astype(np.float32)) * (255.0 if a.max() <= 1.0 else 1.0),
                    0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return np.ascontiguousarray(a[..., :3])


def _jpeg_bytes(img: np.ndarray, max_side: int, quality: int, max_bytes: int
                ) -> Tuple[bytes, int, int]:
    """JPEG no larger than ``max_bytes``: lower the quality first, then the resolution."""
    a = thumbnail(_as_rgb(img), max_side)
    while True:
        for q in (quality, quality - 12, quality - 24, 45):
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="JPEG", quality=q, optimize=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue(), a.shape[1], a.shape[0]
        if max(a.shape[:2]) <= 160:
            return buf.getvalue(), a.shape[1], a.shape[0]
        a = thumbnail(a, int(max(a.shape[:2]) * 0.75))


def _jpeg_uri(img: np.ndarray, max_side: int, quality: int, max_bytes: int
              ) -> Tuple[str, int, int]:
    data, w, h = _jpeg_bytes(img, max_side, quality, max_bytes)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), w, h


def _hex_rgb(c: str) -> Tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _heat_uri(heat: np.ndarray, color: Optional[str] = None) -> str:
    """A localisation map as a small indexed PNG whose palette carries both the colour
    and the transparency (``color=None``: the inferno ramp of the fused map). Maps are
    low-frequency, so a few kilobytes are enough and the browser upscales smoothly."""
    import cv2

    h = np.asarray(heat, dtype=np.float32)
    step = max(1, max(h.shape[:2]) // (2 * HEAT_MAX_SIDE))     # cheap pre-decimation
    h = np.clip(np.nan_to_num(h[::step, ::step]), 0.0, 1.0)
    s = HEAT_MAX_SIDE / max(h.shape[:2])
    if s < 1:
        h = cv2.resize(h, (max(1, int(h.shape[1] * s)), max(1, int(h.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    idx = np.round(h * (HEAT_LEVELS - 1)).astype(np.uint8)
    ramp = np.linspace(0.0, 1.0, HEAT_LEVELS)
    if color is None:
        pal = colormap(ramp)
    else:
        pal = np.tile(np.array(_hex_rgb(color), np.uint8), (HEAT_LEVELS, 1))
    alpha = np.round(255 * ramp ** 0.8).astype(np.uint8)
    im = Image.fromarray(idx)                       # "L"; attaching a palette makes it "P"
    im.putpalette(pal.astype(np.uint8).tobytes())
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True, transparency=bytes(alpha.tolist()))
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _heat_words(ctx: _Ctx, heat: np.ndarray) -> Optional[Tuple[str, str]]:
    """('3.2', 'bottom right') for the strongly highlighted part of a map, or None."""
    m = np.asarray(heat) >= 0.5
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return (f"{100.0 * m.mean():.1f}",
            _position(ctx, xs.mean() / m.shape[1], ys.mean() / m.shape[0]))


def _position(ctx: _Ctx, fx: float, fy: float) -> str:
    grid = ctx.t("pos")
    return grid[min(2, max(0, int(fy * 3)))][min(2, max(0, int(fx * 3)))]


# --------------------------------------------------------------------------- #
# Small components
# --------------------------------------------------------------------------- #
def _bar(value: float, label: str, ticks: Sequence[float] = (), hot: bool = False) -> str:
    """A 0..1 meter with optional threshold marks."""
    v = float(np.clip(value, 0.0, 1.0))
    marks = "".join(f'<u style="inset-inline-start:calc({t * 100:.0f}% - 1px)"></u>'
                    for t in ticks)
    return (f'<div class="bar" role="meter" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{v * 100:.0f}" aria-label="{_esc(label)}">'
            f'<i class="{"hot" if hot else ""}" style="inline-size:{v * 100:.1f}%"></i>'
            f'{marks}</div>')


def _decimate(x: np.ndarray, y: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the minimum and maximum of each bucket, so spikes survive."""
    n = len(x)
    if n <= limit:
        return x, y
    edges = np.linspace(0, n, limit // 2 + 1).astype(int)
    keep: List[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        if not np.isfinite(seg).any():
            keep.append(a)
            continue
        lo, hi = a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))
        keep.extend(sorted({lo, hi}))
    k = np.asarray(keep)
    return x[k], y[k]


def _plot(ctx: _Ctx, name: str, s: Dict[str, Any], duration: float) -> str:
    """One series as an inline SVG line with marker lines; axis text is HTML so that it
    stays readable at any width. Time series share the 0..duration axis of the timeline."""
    try:
        x = np.asarray(s.get("x", []), dtype=np.float64)
        y = np.asarray(s.get("y", []), dtype=np.float64)
    except (TypeError, ValueError):
        return ""
    if x.ndim != 1 or x.shape != y.shape:
        return ""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    fin = np.isfinite(y)
    if fin.sum() < 2:
        return ""
    xlabel, ylabel = str(s.get("xlabel", "")), str(s.get("ylabel", ""))
    is_time = xlabel.lower().startswith("time") and duration > 0
    x0, x1 = (0.0, max(duration, float(x.max()))) if is_time else (float(x.min()), float(x.max()))
    y0, y1 = float(y[fin].min()), float(y[fin].max())
    if x1 - x0 < 1e-12:
        return ""
    if y1 - y0 < 1e-12:
        y0, y1 = y0 - 1.0, y1 + 1.0
    pad = 0.06 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad
    x, y = _decimate(x, y, PLOT_MAX_POINTS)
    px = (x - x0) / (x1 - x0) * 1000.0
    py = (y1 - y) / (y1 - y0) * 100.0
    paths, run = [], []
    for a, b in zip(px, py):
        if np.isfinite(b):
            run.append(f"{a:.1f},{b:.1f}")
        elif run:
            paths.append(run)
            run = []
    if run:
        paths.append(run)
    lines = "".join(f'<path class="ln" d="M{" L".join(p)}"/>' if len(p) > 1 else
                    f'<path class="ln" d="M{p[0]} L{p[0]}" stroke-linecap="round"/>'
                    for p in paths)
    marks = []
    for m in s.get("markers", []) or []:
        try:
            m = float(m)
        except (TypeError, ValueError):
            continue
        if np.isfinite(m) and x0 <= m <= x1:
            marks.append(m)
    mk = "".join(f'<line class="mk" x1="{(m - x0) / (x1 - x0) * 1000:.1f}" y1="0" '
                 f'x2="{(m - x0) / (x1 - x0) * 1000:.1f}" y2="100"/>' for m in marks)
    unit = "s" if is_time else ""
    alt = ctx.t("plot_alt", y=ylabel or name, x=xlabel or "x", x0=_num(x0), x1=_num(x1),
                y0=_num(y0), y1=_num(y1))
    cap = ""
    if marks:
        shown = ", ".join(_num(m) + (" s" if is_time else "") for m in marks[:12])
        shown += " …" if len(marks) > 12 else ""
        alt += " " + ctx.t("plot_marks", list=shown)
        cap = (f'<div class="mkcap" dir="ltr"><i></i>{_esc(ctx.t("marked"))}: '
               f'{_esc(shown)}</div>')
    return (f'<figure class="plot"><figcaption><b>{_ltr(name.replace("_", " "))}</b>'
            f'{_ltr(ylabel)}</figcaption><div class="pbox" dir="ltr">'
            f'<span class="y hi">{_esc(_num(y1))}</span><span class="y lo">{_esc(_num(y0))}</span>'
            f'<svg class="psvg" viewBox="0 0 1000 100" preserveAspectRatio="none" role="img" '
            f'aria-label="{_esc(alt)}" data-x0="{x0:.6g}" data-x1="{x1:.6g}" data-y0="{y0:.6g}" '
            f'data-y1="{y1:.6g}" data-xu="{unit}">{mk}{lines}'
            f'<line class="cur" x1="0" y1="0" x2="0" y2="100"/></svg>'
            f'<div class="px-axis"><span>{_esc(_num(x0))}</span><span>{_esc(xlabel)}</span>'
            f'<span>{_esc(_num(x1))}</span></div>{cap}</div></figure>')


def _ticks(duration: float) -> List[float]:
    """At most about 9 axis ticks for any duration. A file can declare an absurd duration
    (10^12 s was seen), so the step grows with it instead of stopping at one hour."""
    if not np.isfinite(duration) or duration <= 0:
        return []
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if duration / step <= 8:
            break
    else:
        need = duration / 8 / 3600.0                       # hours per tick, at least
        mag = 10.0 ** np.floor(np.log10(need))
        step = 3600.0 * mag * next(m for m in (1, 2, 5, 10) if m * mag >= need)
    out, v = [], 0.0
    while v < duration - 0.6 * step:
        out.append(v)
        v += step
    return out


def _tick_label(v: float, duration: float) -> str:
    return _num(v) if duration < 60 else _clock(v, duration)


def _seg_bars(segs: Sequence[Segment], duration: float, numbered: bool) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        a = float(np.clip(s.start / duration, 0, 1)) * 100
        w = float(np.clip((s.end - s.start) / duration, 0, 1 - a / 100)) * 100
        tip = f"{_clock(s.start, duration)} – {_clock(s.end, duration)}  {s.label}"
        num = f"<b>{i}</b>" if numbered else ""
        out.append(f'<span class="seg" style="left:{a:.3f}%;width:{w:.3f}%" '
                   f'title="{_esc(tip)}">{num}</span>')
    return "".join(out)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _facts(ctx: _Ctx, r: Report, sha256: Optional[str], size: Optional[int],
           generated: _dt.datetime) -> str:
    t = ctx.t
    kind = t("media").get(r.media_type, r.media_type)
    fmt = r.info.get("format") or r.info.get("container_codec") or r.info.get("codec")
    dims = ""
    if r.info.get("width") and r.info.get("height"):
        dims = f'{r.info["width"]}×{r.info["height"]} px'
    dur = _duration(r)
    type_bits = [str(fmt)] if fmt else []
    if dims:
        type_bits.append(dims)
    if dur:
        type_bits.append(_clock(dur, dur) if dur < 60 else _clock(dur, dur) + " min")
    type_html = _esc(kind) + (" · " + _ltr(", ".join(type_bits)) if type_bits else "")
    sha_html = (f'<code class="hash mono" dir="ltr">{_esc(sha256)}</code>' if sha256 else
                f'<span class="muted">{_esc(t("sha_missing"))}</span>')
    size_html = (_ltr(f"{size:,}", "nw") + " " + _esc(t("bytes")) + " (" + _ltr(_human_size(size), "nw") + ")"
                 if size is not None else "–")
    stamp = generated.strftime("%Y-%m-%d %H:%M:%S ") + (generated.strftime("UTC%z") or "")
    rows = [
        (t("file"), f'<span class="name">{_ltr(os.path.basename(r.path) or r.path)}</span>'),
        (t("sha256"), sha_html),
        (t("size"), size_html),
        (t("type"), type_html),
        (t("generated"), f'<time datetime="{_esc(generated.isoformat(timespec="seconds"))}">'
                         f'{_ltr(stamp.strip())}</time>'),
        (t("tool"), _ltr(f"Basira {r.version}".strip(), "nw")
         + (f' · {_esc(t("elapsed"))} {_ltr(f"{r.elapsed:.1f} s", "nw")}' if r.elapsed else "")),
    ]
    wide = ' class="wide"'      # file name and hash span the full width
    body = "".join(f'<dt>{_esc(k)}</dt><dd{wide if i < 2 else ""}>{v}</dd>'
                   for i, (k, v) in enumerate(rows))
    return (f'<section class="card" aria-labelledby="facts-h"><{ctx.h(2)} id="facts-h" '
            f'class="over" style="margin:0 0 8px">{_esc(t("facts"))}</{ctx.h(2)}>'
            f'<dl class="facts">{body}</dl></section>')


def _finding_item(ctx: _Ctx, line: str, titles: Dict[str, Evidence]) -> str:
    """One line of ``Report.explanation``: leading ``[detector]`` tags become labels, the
    sentence itself stays in English unless it is one of the fusion's own sentences."""
    tags = []
    rest = line
    while True:
        m = re.match(r"\s*\[([^\]]+)\]\s*", rest)
        if not m:
            break
        tags.append(m.group(1))
        rest = rest[m.end():]
    labels = []
    for tag in tags:
        if tag.endswith(" track"):
            labels.append(_esc(ctx.t("track")) if tag == "audio track" else _ltr(tag))
        elif tag in titles:
            name, en = ctx.detector_name(titles[tag])
            labels.append(_esc(name) if en or not ctx.rtl else _ltr(name))
        else:
            labels.append(_ltr(tag))
    known = ctx.t("fusion").get(rest.strip()) if ctx.lang != "en" else None
    body = _esc(known) if known else (_say(rest) if ctx.rtl or tags else _esc(rest))
    head = f'<span class="src">{" · ".join(labels)}</span> ' if labels else ""
    return f"<li>{head}{body}</li>"


def _verdict(ctx: _Ctx, r: Report, anchor: str) -> str:
    t = ctx.t
    level = int(np.clip(r.verdict_level, 0, 4))
    key = _verdict_key(r)
    label, means, not_means = t("v")[key]
    known = r.verdict in VERDICT_KEYS
    if ctx.lang == "en" or not known or label is None:
        # English, or wording this module does not know: show the engine's own text.
        vtext = _esc(r.verdict) if ctx.lang == "en" else _ltr(r.verdict)
        vsub = ""
    else:
        vtext = _esc(label)
        vsub = f'<div class="vsub">{_ltr(r.verdict)}</div>'
    current = ' aria-current="true"'
    steps = "".join(f'<li{current if i == level else ""}><b>{i}</b>{_esc(s)}</li>'
                    for i, s in enumerate(t("steps")))
    out = [f'<section class="card verdict lv{level}" id="{anchor}" aria-labelledby="{anchor}-h">'
           f'<div class="vhead"><div class="vicon" aria-hidden="true"><svg viewBox="0 0 48 48">'
           f'{_ICONS[level]}</svg></div><div style="flex:1;min-width:0">'
           f'<{ctx.h(2)} id="{anchor}-h" class="over" style="margin:0">{_esc(t("verdict"))}'
           f'</{ctx.h(2)}><p class="vtext" data-verdict-level="{level}">{vtext}</p>{vsub}'
           f'<ol class="scale" start="0" aria-label="{_esc(t("scale"))}">{steps}</ol>'
           f'</div></div>'
           f'<div class="mean"><div><{ctx.h(3)}>{_esc(t("means"))}</{ctx.h(3)}>'
           f'<p>{_esc(means)}</p></div><div class="not"><{ctx.h(3)}>{_esc(t("not_means"))}'
           f'</{ctx.h(3)}><p>{_esc(not_means)}</p></div></div>']

    out.append(f'<{ctx.h(3)} class="sub">{_esc(t("scores"))}</{ctx.h(3)}><div class="scores">')
    for cat in ("manipulation", "ai_generation", "provenance"):
        v = float(np.clip(r.scores.get(cat, 0.0), 0.0, 1.0))
        name = t("cat_score")[cat]
        ticks = (0.2, 0.45, 0.75) if cat != "provenance" else ()
        out.append(f'<div class="sc"><div class="lab"><span>{_esc(name)}</span>'
                   f'<b>{_ltr(f"{v * 100:.0f}%")}</b></div>'
                   f'{_bar(v, f"{name} {v * 100:.0f}%", ticks, hot=v >= 0.45 and bool(ticks))}'
                   f'</div>')
    out.append("</div>")
    if ctx.depth == 0:
        out.append(f'<p class="small muted" style="margin-top:10px">{_esc(t("scores_help"))} '
                   f'{_esc(t("prov_help"))}</p>')

    if r.explanation:
        titles = {e.title: e for e in r.evidences}
        for c in r.children:
            titles.update({e.title: e for e in c.evidences})
        items = "".join(_finding_item(ctx, x, titles) for x in r.explanation)
        note = (f' <span class="small muted" style="font-weight:400">'
                f'{_esc(t("findings_note"))}</span>') if any("[" in x for x in r.explanation) else ""
        out.append(f'<{ctx.h(3)} class="sub">{_esc(t("findings"))}{note}</{ctx.h(3)}>'
                   f'<ul class="why">{items}</ul>')
    out.append("</section>")
    return "".join(out)


def _layer_color(det: str, used: Dict[str, str]) -> str:
    if det in used:
        return used[det]
    taken = set(used.values())
    slot = LAYER_SLOT.get(det)
    color = LAYER_COLORS[slot] if slot is not None and LAYER_COLORS[slot] not in taken else None
    if color is None:
        color = next((c for c in LAYER_COLORS if c not in taken), LAYER_COLORS[-1])
    used[det] = color
    return color


def _viewer(ctx: _Ctx, r: Report, layer_ids: Dict[str, str]) -> str:
    """The analysed image (or key frame) with switchable localisation maps and boxes.
    ``layer_ids`` is filled with detector id -> checkbox id for "show on image" links."""
    t = ctx.t
    if r.preview is None or r.media_type == "audio":
        return ""
    vid = ctx.uid("viewer")
    src, pw, ph = _jpeg_uri(r.preview, BASE_MAX_SIDE, BASE_QUALITY, BASE_MAX_BYTES)
    full_w = int(r.info.get("width") or 0) or pw
    full_h = int(r.info.get("height") or 0) or ph
    colors: Dict[str, str] = {}

    # ---- map layers: fused first, then detectors that fired, then quiet ones ----
    layers: List[Tuple[str, str, str, str, bool, bool]] = []   # id, label html, uri, alt, on, quiet
    show_fused = (r.fused_heatmap is not None and float(r.fused_heatmap.max()) > 0.05
                  and r.scores.get("manipulation", 0.0) >= 0.2)
    if show_fused:
        words = _heat_words(ctx, r.fused_heatmap)
        alt = t("map_alt", pct=words[0], pos=words[1]) if words else \
            t("map_alt_none", name=t("fused"))
        layers.append(("fused", f'<span class="sw ramp" title="{_esc(t("fused_legend"))}"></span>'
                                f'<span>{_esc(t("fused"))}</span>',
                       _heat_uri(r.fused_heatmap), alt, True, False))
    with_map = [e for e in r.evidences if e.heatmap is not None and not e.error
                and e.confidence > 0 and float(e.heatmap.max()) > 0.05]
    with_map.sort(key=lambda e: (_state(e) != "fired", -_weight(e), -e.score))
    for e in with_map[:MAX_LAYERS - len(layers)]:
        quiet = _state(e) != "fired"
        c = _layer_color(e.detector, colors)
        name, _ = ctx.detector_name(e)
        words = _heat_words(ctx, e.heatmap)
        alt = t("map_alt_layer", name=name, pct=words[0], pos=words[1]) if words else \
            t("map_alt_none", name=name)
        tail = f' <span class="muted">({_esc(t("below"))})</span>' if quiet else ""
        layers.append((e.detector, f'<span class="sw" style="--c:{c}"></span>'
                                   f'<span>{_name_html(ctx, e)}{tail}</span>',
                       _heat_uri(e.heatmap, c), alt, False, quiet))

    # ---- boxes of detectors that fired ----
    shown_frame = next((e.details.get("best_frame") for e in r.evidences
                        if e.heatmap is not None and isinstance(e.details, dict)
                        and e.details.get("best_frame") is not None), None)
    boxes, items = [], []
    fs = max(full_w, full_h) / 42.0
    n = 0
    for e in sorted(r.evidences, key=lambda e: -_weight(e)):
        if _state(e) != "fired":
            continue
        for g in e.regions[:6]:
            n += 1
            c = _layer_color(e.detector, colors)
            bid = f"{vid}-b{n}"
            drawable = g.frame is None or g.frame == shown_frame
            if drawable and g.w > 0 and g.h > 0:
                tx, ty = g.x + fs * 0.75, max(g.y - fs * 0.2, fs * 0.75)
                boxes.append(
                    f'<g class="box" id="{bid}" style="--c:{c}"><rect class="halo" x="{g.x}" '
                    f'y="{g.y}" width="{g.w}" height="{g.h}"/><rect class="edge" x="{g.x}" '
                    f'y="{g.y}" width="{g.w}" height="{g.h}"/><circle class="tag" cx="{tx:.0f}" '
                    f'cy="{ty:.0f}" r="{fs * 0.72:.0f}"/><text x="{tx:.0f}" '
                    f'y="{ty + fs * 0.36:.0f}" font-size="{fs:.0f}">{n}</text></g>')
            pos = _position(ctx, (g.x + g.w / 2) / max(full_w, 1), (g.y + g.h / 2) / max(full_h, 1))
            words = t("region_words", pos=pos, w=int(g.w), h=int(g.h), x=int(g.x), y=int(g.y),
                      pct=f"{100.0 * g.w * g.h / max(full_w * full_h, 1):.1f}")
            if g.frame is not None:
                words += " · " + t("region_frame", n=int(g.frame))
            lab = f'{_say(g.label)}' if g.label else ""
            items.append(f'<li data-box="{bid}"><span class="num" style="--c:{c}">{n}</span>'
                         f'<span><b>{_name_html(ctx, e)}</b>{lab}'
                         f'<span class="muted">{_esc(words)}</span></span></li>')

    # ---- markup ----
    imgs = []
    toggles = []
    for i, (det, label, uri, alt, default_on, quiet) in enumerate(layers):
        cid = f"{vid}-l{i}"
        layer_ids[det] = cid
        imgs.append(f'<img class="layer l{i}{" dflt" if default_on else ""}" src="{uri}" '
                    f'alt="{_esc(alt)}">')
        toggles.append(f'<label class="opt"><input type="checkbox" class="lt{i}" id="{cid}"'
                       f'{" checked" if default_on else ""}>{label}</label>')
    if boxes:
        toggles.append(f'<label class="opt"><input type="checkbox" class="ltb" checked>'
                       f'<span>{_esc(t("boxes"))}</span></label>')
    svg = (f'<svg class="boxes" viewBox="0 0 {full_w} {full_h}" preserveAspectRatio="none" '
           f'aria-hidden="true">{"".join(boxes)}</svg>') if boxes else ""
    alt_base = t("alt_video") if r.media_type == "video" else t("alt_image")
    max_w = max(pw, 420)
    stage = (f'<div class="stage-scroll" tabindex="0" role="group" '
             f'aria-label="{_esc(t("scroll_label"))}"><div class="stage" '
             f'style="max-width:min({max_w}px,calc(78vh*{pw / max(ph, 1):.4f}))">'
             f'<img class="{"px" if pw < 240 else ""}" src="{src}" width="{pw}" height="{ph}" '
             f'alt="{_esc(alt_base)}"><div class="ov">{"".join(imgs)}{svg}</div>'
             f'<div class="divider"></div></div></div>')

    side = []
    if toggles:
        side.append(f'<fieldset><legend>{_esc(t("layers"))}</legend>{"".join(toggles)}</fieldset>')
    if layers:
        side.append(f'<div class="ctl js-only"><label for="{vid}-op">{_esc(t("opacity"))}</label>'
                    f'<input type="range" class="op" id="{vid}-op" dir="ltr" min="0" max="100" value="80">'
                    f'</div>')
    if layers or boxes:
        side.append(f'<div class="ctl js-only"><label for="{vid}-sp">{_esc(t("compare"))}</label>'
                    f'<input type="range" class="split" id="{vid}-sp" dir="ltr" min="0" max="100" value="0" '
                    f'aria-describedby="{vid}-sph"><span class="small muted" id="{vid}-sph">'
                    f'{_esc(t("compare_hint"))}</span></div>')
    side.append(f'<div class="ctl js-only"><button type="button" class="btn zoom-btn" '
                f'aria-pressed="false">{_esc(t("zoom"))}</button><span class="small muted" '
                f'style="display:block;margin-top:4px">{_esc(t("zoom_hint"))}</span></div>')
    if items:
        side.append(f'<div><{ctx.h(3)} style="font-size:13px;margin-bottom:4px">'
                    f'{_esc(t("regions"))}</{ctx.h(3)}><ol class="rlist">{"".join(items)}</ol></div>')
    notes = []
    if r.media_type == "video":
        notes.append(t("keyframe_note"))
    if not show_fused:
        notes.append(t("no_map"))
    note = f'<p class="small muted" style="margin:10px 0 0">{_esc(" ".join(notes))}</p>' \
        if notes else ""
    title = t("where_video") if r.media_type == "video" else t("where_image")
    return (f'<section id="{vid}" aria-labelledby="{vid}-h"><{ctx.h(2)} id="{vid}-h">'
            f'{_esc(title)}</{ctx.h(2)}><div class="card"><div class="viewer" dir="ltr">'
            f'{stage}<div class="vside" dir="{"rtl" if ctx.rtl else "ltr"}">{"".join(side)}'
            f'</div></div>{note}</div></section>')


def _timeline(ctx: _Ctx, r: Report) -> str:
    """Spectrogram (audio), one lane for the merged ranges and one per detector, on a
    shared time axis, followed by the same ranges as a table."""
    t = ctx.t
    dur = _duration(r)
    if r.media_type not in ("audio", "video") or dur <= 0:
        return ""
    tid = ctx.uid("timeline")
    segs = [s for s in r.fused_segments if s.end >= 0 and s.start <= dur]
    parts = []
    if r.media_type == "audio" and r.preview is not None:
        src, _, _ = _jpeg_uri(r.preview, 1200, 78, VISUAL_MAX_BYTES)
        nyq = r.info.get("sample_rate")
        hi = f'<span class="hi">{_esc(_num(float(nyq) / 2000.0))} kHz</span>' if nyq else ""
        parts.append(f'<div class="specbox"><img class="spec zoomable" src="{src}" '
                     f'alt="{_esc(t("alt_audio"))}" data-cap="{_esc(t("spectrogram"))}">{hi}'
                     f'<span class="lo">0 Hz</span></div>')
    parts.append(f'<div class="lane all"><span class="nm" dir="{"rtl" if ctx.rtl else "ltr"}">'
                 f'{_esc(t("lane_all"))}</span><div class="track">'
                 f'{_seg_bars(segs, dur, True)}</div></div>')
    for e in sorted(r.evidences, key=lambda e: -_weight(e)):
        if _state(e) == "fired" and e.segments:
            parts.append(f'<div class="lane"><span class="nm" dir="{"rtl" if ctx.rtl else "ltr"}">'
                         f'{_name_html(ctx, e)}</span><div class="track">'
                         f'{_seg_bars(e.segments, dur, False)}</div></div>')
    ticks = "".join(f'<span style="left:{v / dur * 100:.2f}%">{_esc(_tick_label(v, dur))}</span>'
                    for v in _ticks(dur))
    ticks += f'<span class="end" style="left:100%">{_esc(_clock(dur, dur))}</span>'
    parts.append(f'<div class="axis">{ticks}</div>')
    alt = t("tl_alt", dur=_clock(dur, dur), n=len(segs))
    if segs:
        rows = "".join(
            f'<tr><td class="n" data-l="#"><span class="num">{i}</span></td>'
            f'<td class="n" data-l="{_esc(t("col_from"))}">{_ltr(_clock(s.start, dur))}</td>'
            f'<td class="n" data-l="{_esc(t("col_to"))}">{_ltr(_clock(s.end, dur))}</td>'
            f'<td class="n" data-l="{_esc(t("col_score"))}">{_ltr(f"{s.score:.2f}")}</td>'
            f'<td data-l="{_esc(t("col_what"))}">{_say(s.label)}</td></tr>'
            for i, s in enumerate(segs[:40], 1))
        table = (f'<{ctx.h(3)} class="sub">{_esc(t("ranges"))}</{ctx.h(3)}><div class="tw">'
                 f'<table class="resp"><thead><tr><th scope="col">#</th><th scope="col">'
                 f'{_esc(t("col_from"))}</th><th scope="col">{_esc(t("col_to"))}</th>'
                 f'<th scope="col">{_esc(t("col_score"))}</th><th scope="col">'
                 f'{_esc(t("col_what"))}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        table = f'<p class="muted" style="margin:12px 0 0">{_esc(t("no_ranges"))}</p>'
    return (f'<section id="{tid}" aria-labelledby="{tid}-h"><{ctx.h(2)} id="{tid}-h">'
            f'{_esc(t("timeline"))}</{ctx.h(2)}><div class="card"><div class="tl" dir="ltr" '
            f'role="group" aria-label="{_esc(alt)}">{"".join(parts)}</div>{table}</div></section>')


def _detector_card(ctx: _Ctx, e: Evidence, state: str, duration: float,
                   layer_ids: Dict[str, str], anchor: str) -> str:
    t = ctx.t
    name, en = ctx.detector_name(e)
    head = [f'<{ctx.h(4)} id="{anchor}">{_esc(name) if en or not ctx.rtl else _ltr(name)}'
            f'</{ctx.h(4)}>']
    if en:
        head.append(f'<span class="en">{_ltr(en)}</span>')
    head.append(f'<span class="st {state}">{_esc(t("state")[state])}</span>')
    head.append(f'<span class="chip">{_esc(t("cat").get(e.category.value, e.category.value))}'
                f'</span>')
    if e.decisive and state == "fired":
        head.append(f'<span class="chip dec">{_esc(t("decisive"))}</span>')
    if state == "quiet" and e.confidence < 0.2:
        head.append(f'<span class="chip">{_esc(t("low_conf"))}</span>')
    head.append(f'<code class="muted">{_ltr(e.detector)}</code>')
    out = [f'<article class="card det" aria-labelledby="{anchor}"><div class="dhead">'
           f'{"".join(head)}</div>']

    about = ctx.about(e)
    if about:
        out.append(f'<dl class="about"><dt>{_esc(t("measures"))}</dt><dd>{_esc(about[0])}</dd>'
                   f'<dt>{_esc(t("blind"))}</dt><dd>{_esc(about[1])}</dd></dl>')
    else:
        out.append(f'<p class="small muted">{_esc(t("no_about"))}</p>')

    if state == "failed":
        out.append(f'<div class="errbox"><span class="k small"><b>{_esc(t("error"))}</b></span> '
                   f'{_say(e.error)}</div></article>')
        return "".join(out)

    if state != "na":
        w = _weight(e)
        gauges = [(t("score"), e.score, (0.5,), e.score >= 0.5),
                  (t("confidence"), e.confidence, (), False)]
        if state == "fired":
            gauges.append((t("weight"), w, (), False))
        out.append('<div class="gauges">' + "".join(
            f'<div><div class="lab"><span>{_esc(lab)}</span><b>{_ltr(_score(v))}</b></div>'
            f'{_bar(v, f"{lab} {_score(v)}", ticks, hot)}</div>'
            for lab, v, ticks, hot in gauges) + "</div>")
    out.append(f'<div class="finding"><span class="k">{_esc(t("finding"))}</span>'
               f'{_say(e.summary)}</div>')

    figs = []
    for vname, img in e.visuals.items():
        try:
            cap = max(12_000, min(VISUAL_MAX_BYTES, ctx.visual_bytes))
            uri, w_, h_ = _jpeg_uri(img, VISUAL_MAX_SIDE, VISUAL_QUALITY, cap)
        except Exception:
            continue
        ctx.visual_bytes -= len(uri) * 3 // 4
        cap = vname.replace("_", " ")
        figs.append(f'<figure><img class="zoomable" src="{uri}" width="{w_}" height="{h_}" '
                    f'alt="{_esc(cap)}" data-cap="{_esc(e.title + " — " + cap)}">'
                    f'<figcaption>{_ltr(cap)}</figcaption></figure>')
    if figs:
        out.append(f'<div class="figs">{"".join(figs)}</div>')
    if e.detector in layer_ids:
        target = layer_ids[e.detector].rsplit("-l", 1)[0]
        out.append(f'<p style="margin:12px 0 0"><a class="btn noprint" href="#{target}" '
                   f'data-layer="{layer_ids[e.detector]}">{_esc(t("show_on_image"))}</a></p>')
    for sname, s in e.series.items():
        if isinstance(s, dict):
            out.append(_plot(ctx, str(sname), s, duration))
    if e.segments:
        total = duration or max(s.end for s in e.segments)
        rows = "".join(
            f'<tr><td class="n">{_ltr(_clock(s.start, total))}</td>'
            f'<td class="n">{_ltr(_clock(s.end, total))}</td>'
            f'<td class="n">{_ltr(f"{s.score:.2f}")}</td><td>{_say(s.label)}</td></tr>'
            for s in e.segments[:20])
        out.append(f'<div class="tw" style="margin-top:12px"><table><thead><tr><th scope="col">'
                   f'{_esc(t("col_from"))}</th><th scope="col">{_esc(t("col_to"))}</th>'
                   f'<th scope="col">{_esc(t("col_score"))}</th><th scope="col">'
                   f'{_esc(t("col_what"))}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if e.details:
        raw = json.dumps(e.to_dict()["details"], indent=2, ensure_ascii=False, default=str)
        if len(raw) > RAW_JSON_LIMIT:
            raw = raw[:RAW_JSON_LIMIT] + "\n… " + t("truncated")
        out.append(f'<details class="raw"><summary>{_esc(t("raw"))}</summary>'
                   f'<pre dir="ltr">{_esc(raw)}</pre></details>')
    out.append("</article>")
    return "".join(out)


def _evidence(ctx: _Ctx, r: Report, layer_ids: Dict[str, str],
              anchors_out: Optional[Dict[int, str]] = None) -> str:
    t = ctx.t
    eid = ctx.uid("evidence")
    dur = _duration(r)
    groups: Dict[str, List[Evidence]] = {"fired": [], "failed": [], "quiet": [], "na": []}
    for e in r.evidences:
        groups[_state(e)].append(e)
    groups["fired"].sort(key=lambda e: (not e.decisive, -_weight(e), -e.score))
    groups["quiet"].sort(key=lambda e: -(e.score * e.confidence))
    anchors = {id(e): f"{eid}-{_dom_id(e.detector)}-{i}"
               for i, e in enumerate(r.evidences)}
    if anchors_out is not None:
        anchors_out.update(anchors)

    rows = []
    for state in ("fired", "failed", "quiet", "na"):
        for e in groups[state]:
            num = state in ("fired", "quiet")
            rows.append(
                f'<tr><th scope="row" data-l="{_esc(t("col_detector"))}">'
                f'<a href="#{anchors[id(e)]}">{_name_html(ctx, e)}</a></th>'
                f'<td data-l="{_esc(t("col_category"))}">'
                f'{_esc(t("cat").get(e.category.value, e.category.value))}</td>'
                f'<td data-l="{_esc(t("col_outcome"))}"><span class="st {state}">'
                f'{_esc(t("state")[state])}</span></td>'
                f'<td class="n" data-l="{_esc(t("score"))}">'
                f'{_ltr(_score(e.score)) if num else "–"}</td>'
                f'<td class="n" data-l="{_esc(t("col_conf"))}">'
                f'{_ltr(f"{e.confidence:.2f}") if num else "–"}</td>'
                f'<td data-l="{_esc(t("col_finding"))}">{_say(e.summary)}</td></tr>')
    table = (f'<div class="card"><{ctx.h(3)} style="margin-bottom:6px">{_esc(t("overview"))}'
             f'</{ctx.h(3)}><div class="tw"><table class="resp"><thead><tr>'
             f'<th scope="col">{_esc(t("col_detector"))}</th>'
             f'<th scope="col">{_esc(t("col_category"))}</th>'
             f'<th scope="col">{_esc(t("col_outcome"))}</th>'
             f'<th scope="col">{_esc(t("score"))}</th><th scope="col">{_esc(t("col_conf"))}</th>'
             f'<th scope="col">{_esc(t("col_finding"))}</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div></div>') if rows else ""

    blocks = []
    for state in ("fired", "failed", "quiet", "na"):
        evs = groups[state]
        if not evs:
            continue
        cards = "".join(_detector_card(ctx, e, state, dur, layer_ids, anchors[id(e)])
                        for e in evs)
        is_open = " open" if state in ("fired", "failed") else ""
        blocks.append(
            f'<details class="grp"{is_open}><summary><{ctx.h(3)}><span class="st {state}">'
            f'{_esc(t("state")[state])}</span> {_ltr(f"({len(evs)})")}</{ctx.h(3)}>'
            f'<span class="small muted">{_esc(t("state_help")[state])}</span></summary>'
            f'{cards}</details>')
    return (f'<section id="{eid}" aria-labelledby="{eid}-h"><{ctx.h(2)} id="{eid}-h">'
            f'{_esc(t("evidence"))}</{ctx.h(2)}><p class="lead">{_esc(t("evidence_help"))}</p>'
            f'{table}{"".join(blocks)}</section>')


# --------------------------------------------------------------------------- #
# How the verdict was reached: file -> detectors -> categories -> verdict
# --------------------------------------------------------------------------- #
FLOW_W = 1040
_FLOW_COLS = {"file": (8, 176), "det": (262, 276), "cat": (622, 196), "vd": (862, 170)}
_FLOW_CATS = ("manipulation", "ai_generation", "provenance")
_DET_H, _DET_GAP, _CAT_H, _CAT_GAP, _TOP = 50, 10, 66, 18, 34
_DRIVES = 0.8      # weight of a child report's category scores (fusion.finalize)


def _fit(px: float) -> int:
    """Characters of the bold 12.5 px node title that fit in ``px`` (a safe average)."""
    return max(8, int(px / 7.4))


def _trunc(s: str, n: int) -> str:
    s = _clean(s)
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


class _Flow:
    """SVG writer in left-to-right logical coordinates, mirrored for right-to-left pages,
    so that the diagram reads in the direction of the page. Every string goes through
    :func:`_esc`; every text element holds one script only, so bidi never reorders it."""

    def __init__(self, rtl: bool):
        self.rtl, self.out = rtl, []

    def x(self, x: float) -> float:
        return FLOW_W - x if self.rtl else x

    def rect(self, x, y, w, h, cls, rx=6) -> None:
        x0 = FLOW_W - x - w if self.rtl else x
        self.out.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" '
                        f'height="{h:.1f}" rx="{rx}"/>')

    def text(self, x, y, s, cls="", end=False) -> None:
        """``end`` aligns to the far edge in reading order. Words run in the page's
        direction (so "(ELA)" and "…" sit right in Arabic); numbers stay left-to-right,
        or "+0.80" would read "0.80+"."""
        numeric = bool({"n", "elab"} & set(cls.split()))
        if self.rtl and not numeric:
            anchor, extra = ("end" if end else "start"), ' direction="rtl"'
        else:
            anchor, extra = ("end" if end != self.rtl else "start"), ""
        c = f' class="{cls}"' if cls else ""
        self.out.append(f'<text{c} x="{self.x(x):.1f}" y="{y:.1f}" text-anchor="{anchor}"'
                        f'{extra}>{_esc(s)}</text>')

    def line(self, x1, y1, x2, y2, cls) -> None:
        self.out.append(f'<line class="{cls}" x1="{self.x(x1):.1f}" y1="{y1:.1f}" '
                        f'x2="{self.x(x2):.1f}" y2="{y2:.1f}"/>')

    def bar(self, x, y, w, value, ticks=(0.5,)) -> None:
        v = float(np.clip(value, 0.0, 1.0))
        self.rect(x, y, w, 7, "trk", 3)
        if w * v >= 1.0:
            self.rect(x, y, w * v, 7, "fill", 3)
        for tk in ticks:
            self.line(x + w * tk, y - 3, x + w * tk, y + 10, "tick")

    def edge(self, x1, y1, x2, y2, cls, width, delay, pulse=False) -> None:
        dx = (x2 - x1) * 0.5
        d = (f"M{self.x(x1):.1f},{y1:.1f} C{self.x(x1 + dx):.1f},{y1:.1f} "
             f"{self.x(x2 - dx):.1f},{y2:.1f} {self.x(x2):.1f},{y2:.1f}")
        self.out.append(f'<path class="e {cls} an" d="{d}" style="stroke-width:{width:.2f}px;'
                        f'--d:{delay:.2f}s"/>')
        if pulse:
            self.out.append(f'<path class="pl" d="{d}" pathLength="1" '
                            f'style="stroke-width:{width + 2.5:.2f}px;--d:{delay:.2f}s"/>')

    def raw(self, s: str) -> None:
        self.out.append(s)


def _flow(ctx: _Ctx, r: Report, anchors: Dict[int, str], child_ids: List[str]) -> str:
    """The analysis as a diagram drawn from the report itself: no number in it is made up.
    Static SVG (readable without JavaScript and in print); the script only replays it."""
    t = ctx.t
    if not r.evidences and not r.children:
        return ""
    fid = ctx.uid("flow")
    g = _Flow(ctx.rtl)
    order = {c: i for i, c in enumerate(_FLOW_CATS)}
    evs = sorted(r.evidences, key=lambda e: order.get(
        getattr(e.category, "value", e.category), 9))
    n = len(evs) + len(r.children)
    det_h = n * (_DET_H + _DET_GAP) - _DET_GAP
    cat_h = 3 * _CAT_H + 2 * _CAT_GAP
    body_h = max(det_h, cat_h, 150)
    height = _TOP + body_h + 14

    fx, fw = _FLOW_COLS["file"]
    dx, dw = _FLOW_COLS["det"]
    cx, cw = _FLOW_COLS["cat"]
    vx, vw = _FLOW_COLS["vd"]
    for (x, w), label in ((_FLOW_COLS["file"], t("file")), (_FLOW_COLS["det"], t("col_detector")),
                          (_FLOW_COLS["cat"], t("flow_cats")), (_FLOW_COLS["vd"], t("verdict"))):
        g.text(x + 2, 16, label, "hd")

    # category and verdict geometry first: the edges need their anchor points
    cy0 = _TOP + (body_h - cat_h) / 2
    cat_y = {c: cy0 + i * (_CAT_H + _CAT_GAP) for i, c in enumerate(_FLOW_CATS)}
    scores = {c: float(r.scores.get(c, 0.0) or 0.0) for c in _FLOW_CATS}
    level = int(r.verdict_level)
    decisive = any(e.decisive and e.score >= 0.9 and not e.error for e in r.evidences)
    top_cat = "ai_generation" if scores["ai_generation"] >= scores["manipulation"] \
        else "manipulation"
    vh = 150
    vy = _TOP + (body_h - vh) / 2
    file_h = 96
    file_y = _TOP + (body_h - file_h) / 2

    # incoming edges per category, so they spread over the category's left side
    feeds: Dict[str, List[Tuple[float, str, float, str]]] = {c: [] for c in _FLOW_CATS}
    rows: List[Tuple[str, Any]] = [("ev", e) for e in evs] + [("child", c) for c in r.children]

    groups = []
    for i, (kind, item) in enumerate(rows):
        y = _TOP + i * (_DET_H + _DET_GAP)
        mid = y + _DET_H / 2
        ny = file_y + 16 + (file_h - 32) * (i / max(1, n - 1) if n > 1 else 0.5)
        d_node = 0.6 + i * 0.05
        d_out = 1.3 + i * 0.04
        sub = _Flow(ctx.rtl)
        sub.edge(fx + fw, ny, dx, mid, "in", 1.0, 0.35 + i * 0.03)
        if kind == "ev":
            e = item
            state = _state(e)
            w = _weight(e) if state == "fired" else 0.0
            cat = getattr(e.category, "value", e.category)
            if cat in feeds:
                feeds[cat].append((mid, state, w, f"{d_out:.2f}"))
            href = anchors.get(id(e))
            name = ctx.detector_name(e)[0]
            link_open = (f'<a href="#{href}" tabindex="-1">' if href else "<g>")
            link_close = "</a>" if href else "</g>"
            sub.raw(f'<g class="an" style="--d:{d_node:.2f}s">{link_open}')
            sub.rect(dx, y, dw, _DET_H, "box")
            label = _trunc(t("decisive") if e.decisive and state == "fired"
                           else t("state")[state], 16)
            sub.text(dx + 10, y + 19, _trunc(name, _fit(dw - 30 - 6.2 * len(label))), "nm")
            sub.text(dx + dw - 10, y + 19, label, f"s-{state} sm", end=True)
            if state in ("fired", "quiet"):
                sub.bar(dx + 10, y + 31, 112, e.score)
                sub.text(dx + 130, y + 38.5, _score(e.score), "n")
                sub.text(dx + dw - 10, y + 38.5, f"{e.confidence:.2f}", "n", end=True)
                sub.text(dx + dw - 46, y + 38.5, t("col_conf"), "sm", end=True)
            else:
                sub.text(dx + 10, y + 38.5, _trunc(t("state_help")[state], 44), "sm")
            sub.raw(f"{link_close}</g>")
            groups.append((f"nd {state}", sub, y))
        else:
            c = item
            cs = {k: float(c.scores.get(k, 0.0) or 0.0) for k in _FLOW_CATS}
            lv = int(c.verdict_level)
            state = "fired" if lv >= 2 else "quiet"
            for k in ("manipulation", "ai_generation"):
                w = _DRIVES * cs[k]
                feeds[k].append((mid, "fired" if w >= 0.1 else "quiet", w, f"{d_out:.2f}"))
            kind_name = t("media").get(c.media_type, c.media_type)
            name = t("child", kind=kind_name.lower() if ctx.lang == "en" else kind_name)
            href = child_ids[i - len(evs)] if i - len(evs) < len(child_ids) else ""
            sub.raw(f'<g class="an" style="--d:{d_node:.2f}s">'
                    + (f'<a href="#{href}" tabindex="-1">' if href else "<g>"))
            sub.rect(dx, y, dw, _DET_H, "box")
            label = _trunc(t("steps")[max(0, min(4, lv))], 20)
            sub.text(dx + 10, y + 19, _trunc(name, _fit(dw - 30 - 6.2 * len(label))), "nm")
            sub.text(dx + dw - 10, y + 19, label, f"s-{state} sm", end=True)
            top = max(cs["manipulation"], cs["ai_generation"])
            sub.bar(dx + 10, y + 31, 112, top, ticks=(0.2, 0.45, 0.75))
            sub.text(dx + 130, y + 38.5, f"{top * 100:.0f}%", "n")
            sub.raw("</a>" if href else "</g>")
            sub.raw("</g>")
            groups.append((f"nd {state}", sub, y))

    # detector -> category edges belong to the detector's group (hover isolates them)
    gi = 0
    for kind, item in rows:
        cls, sub, y = groups[gi]
        mid = y + _DET_H / 2
        cats = ([getattr(item.category, "value", item.category)] if kind == "ev"
                else ["manipulation", "ai_generation"])
        for cat in cats:
            if cat not in feeds:
                continue
            ins = feeds[cat]
            j = next(k for k, f in enumerate(ins) if f[0] == mid)
            _, state, w, delay = ins[j]
            ty = cat_y[cat] + 12 + (_CAT_H - 24) * ((j + 0.5) / len(ins))
            width = 1.4 + 7.0 * w if w > 0 else 1.0
            sub.edge(dx + dw, mid, cx, ty, state, width, float(delay),
                     pulse=state == "fired" and w > 0)
            if w >= 0.005:
                sub.raw(f'<g class="an" style="--d:{delay}s">')
                sub.text(dx + dw + 8, mid - 6, f"+{w:.2f}", "elab")
                sub.raw("</g>")
        gi += 1

    for cls, sub, _ in groups:
        g.raw(f'<g class="{cls}">' + "".join(sub.out) + "</g>")

    # categories
    for i, cat in enumerate(_FLOW_CATS):
        y = cat_y[cat]
        v = scores[cat]
        hot = cat == top_cat and level >= 1 and v >= 0.2 and not decisive
        cls = "cat" + (" hot" if hot else "") + (" prov" if cat == "provenance" else "")
        g.raw(f'<g class="{cls} an" style="--d:1.9s">')
        g.rect(cx, y, cw, _CAT_H, "box")
        g.text(cx + 10, y + 20, _trunc(t("cat_score")[cat], 26), "nm")
        g.text(cx + cw - 10, y + 44, f"{v * 100:.0f}%", "big n", end=True)
        g.bar(cx + 10, y + 46, cw - 70, v, ticks=(0.2, 0.45, 0.75))
        g.raw("</g>")
        ecls = "hot" if hot else ("prov" if cat == "provenance" else "")
        ty = vy + 40 + i * 30
        g.edge(cx + cw, y + _CAT_H / 2, vx, ty, ecls, 1.4 + 6.0 * v if v > 0 else 1.0,
               2.3, pulse=hot)

    # the file
    g.raw('<g class="an" style="--d:0s">')
    g.rect(fx, file_y, fw, file_h, "box")
    g.text(fx + 10, file_y + 22, t("media").get(r.media_type, r.media_type), "nm")
    g.text(fx + 10, file_y + 44, _trunc(os.path.basename(str(r.path)) or str(r.path), 24), "sm")
    g.text(fx + 10, file_y + 72, t("col_detector"), "sm")
    g.text(fx + fw - 10, file_y + 72, str(len(r.evidences)), "n", end=True)
    g.raw("</g>")

    # the verdict and its five-step scale
    vk = f"lv{max(0, min(4, level))}"
    g.raw(f'<g class="vd {vk} an" style="--d:2.7s">')
    g.rect(vx, vy, vw, vh, "box")
    g.text(vx + 12, vy + 24, t("verdict"), "hd")
    g.text(vx + 12, vy + 58, _trunc(t("steps")[max(0, min(4, level))], 20), "big")
    sw = (vw - 24 - 4 * 4) / 5
    for k in range(5):
        g.rect(vx + 12 + k * (sw + 4), vy + 80, sw, 10, "stp on" if k == level else "stp", 2)
    g.text(vx + 12, vy + 116, t("cat_score")[top_cat] if not decisive else t("decisive"),
           "sm")
    g.text(vx + vw - 12, vy + 136, f"{scores[top_cat] * 100:.0f}%" if not decisive else "",
           "n", end=True)
    g.raw("</g>")

    svg = (f'<svg class="fsvg" viewBox="0 0 {FLOW_W} {height:.0f}" aria-hidden="true" '
           f'focusable="false" preserveAspectRatio="xMidYMin meet">{"".join(g.out)}</svg>')

    # the same path in words: the text alternative, and the chapters of the replay
    st = {k: 0 for k in ("fired", "quiet", "na", "failed")}
    for e in r.evidences:
        st[_state(e)] += 1
    name = os.path.basename(str(r.path)) or str(r.path)
    steps = [
        (0.0, f'{_esc(t("flow_s1"))}: {_ltr(name)}'),
        (0.6, _esc(t("flow_s2", total=len(r.evidences), **st))
         + (" " + _esc(t("flow_s2_track")) if r.children else "")),
        (1.3, _esc(t("flow_s3", m=f"{scores['manipulation'] * 100:.0f}",
                     a=f"{scores['ai_generation'] * 100:.0f}",
                     p=f"{scores['provenance'] * 100:.0f}"))),
        (2.1, _esc(t("flow_s4_dec") if decisive else
                   t("flow_s4", cat=t("cat_score")[top_cat], v=f"{scores[top_cat] * 100:.0f}"))),
        (2.7, f'<b>{_esc(t("flow_s5"))}:</b> {_esc(t("v")[_verdict_key(r)][0] or r.verdict)}'),
    ]
    items = "".join(f'<li style="--d:{d:.1f}s">{s}</li>' for d, s in steps)
    key = (f'<ul class="fkey"><li><i class="k-w"></i>{_esc(t("weight"))}</li>'
           f'<li><i class="k-quiet"></i>{_esc(t("state")["quiet"])}</li>'
           f'<li><i class="k-na"></i>{_esc(t("state")["na"])}</li>'
           f'<li><i class="k-failed"></i>{_esc(t("state")["failed"])}</li></ul>')
    return (f'<section id="{fid}" aria-labelledby="{fid}-h"><div class="fhead">'
            f'<{ctx.h(2)} id="{fid}-h">{_esc(t("flow"))}</{ctx.h(2)}>'
            f'<button type="button" class="btn js-only replay">{_esc(t("flow_replay"))}</button>'
            f'</div><p class="lead">{_esc(t("flow_help"))}</p>'
            f'<div class="card flow"><div class="fscroll" dir="ltr">{svg}</div>{key}'
            f'<ol class="fsteps">{items}</ol></div></section>')


def _file_details(ctx: _Ctx, r: Report) -> str:
    t = ctx.t
    rows = []
    for k, v in r.info.items():
        if v is None or isinstance(v, (dict, np.ndarray)) or (
                isinstance(v, (str, list, tuple)) and len(v) == 0):
            continue
        if k in ("detectors_failed", "children_failed") and not v:
            continue
        if isinstance(v, (list, tuple)):
            v = "; ".join(str(x) for x in v)
        elif isinstance(v, (float, np.floating)):
            v = f"{float(v):.6g}"
        rows.append(f'<tr><th scope="row">{_ltr(str(k).replace("_", " "))}</th>'
                    f'<td>{_ltr(v)}</td></tr>')
    if not rows:
        return ""
    did = ctx.uid("details")
    return (f'<section id="{did}" aria-labelledby="{did}-h"><{ctx.h(2)} id="{did}-h">'
            f'{_esc(t("details"))}</{ctx.h(2)}><p class="lead">{_esc(t("details_help"))}</p>'
            f'<div class="card"><div class="tw"><table class="kv"><tbody>{"".join(rows)}'
            f'</tbody></table></div></div></section>')


def _body(ctx: _Ctx, r: Report, toc: List[Tuple[str, str]]) -> str:
    """Verdict, localisation, timeline, evidence and file details of one report; the
    same function renders an embedded track one heading level down."""
    t = ctx.t
    anchor = "verdict" if ctx.depth == 0 else ctx.uid("verdict")
    layer_ids: Dict[str, str] = {}
    viewer = _viewer(ctx, r, layer_ids)
    timeline = _timeline(ctx, r)
    anchors: Dict[int, str] = {}
    evidence = _evidence(ctx, r, layer_ids, anchors)
    child_ids = [ctx.uid("track") for _ in r.children]
    flow = _flow(ctx, r, anchors, child_ids) if ctx.depth == 0 else ""
    details = _file_details(ctx, r)
    if ctx.depth == 0:
        toc.append((anchor, t("verdict")))
        for chunk, label in ((viewer, t("where_video") if r.media_type == "video"
                              else t("where_image")), (timeline, t("timeline")),
                             (flow, t("flow")), (evidence, t("evidence")),
                             (details, t("details"))):
            m = re.match(r'<section id="([^"]+)"', chunk)
            if m:
                toc.append((m.group(1), label))
    out = [_verdict(ctx, r, anchor), viewer, timeline, flow, evidence, details]
    for c, cid in zip(r.children, child_ids):
        kind = t("media").get(c.media_type, c.media_type)
        title = t("child", kind=kind.lower() if ctx.lang == "en" else kind)
        toc.append((cid, title))
        ctx.depth += 1
        inner = _body(ctx, c, toc)
        ctx.depth -= 1
        out.append(f'<section id="{cid}" aria-labelledby="{cid}-h"><{ctx.h(2)} id="{cid}-h">'
                   f'{_esc(title)}</{ctx.h(2)}><p class="lead">{_esc(t("child_note"))}</p>'
                   f'<div class="child">{inner}</div></section>')
    return "".join(out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _lang(lang: Optional[str]) -> str:
    code = (lang or "en").replace("_", "-").split("-")[0].lower()
    if code not in LANGS:
        raise ValueError(f"unsupported report language {lang!r}; choose one of {LANGS}")
    return code


def render_html(report: Report, lang: str = "en", *, sha256: Optional[str] = None,
                file_size: Optional[int] = None,
                generated: Optional[_dt.datetime] = None) -> str:
    """Render ``report`` as one self-contained HTML document.

    ``lang``       ``"en"`` or ``"ar"`` (right-to-left interface).
    ``sha256``     hex digest of the analysed file. If omitted it is computed from
                   ``report.path`` when that file still exists; pass it explicitly when
                   the file is gone by the time the report is written (uploads).
    ``file_size``  size in bytes, same fallback as ``sha256``.
    ``generated``  timestamp printed in the header (default: now, local time zone).
    """
    r = report
    ctx = _Ctx(_lang(lang))
    t = ctx.t
    on_disk = isinstance(r.path, str) and os.path.isfile(r.path)
    if sha256 is None and on_disk:
        sha256 = _sha256_of(r.path)
    if sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(sha256)):
        raise ValueError("sha256 must be a 64-character hex digest")
    if file_size is None:
        if on_disk:
            file_size = os.path.getsize(r.path)
        else:
            raw = r.info.get("bytes", r.info.get("file_size"))
            file_size = int(raw) if isinstance(raw, (int, np.integer)) else None
    generated = generated or _dt.datetime.now().astimezone()

    toc: List[Tuple[str, str]] = []
    body = _body(ctx, r, toc)
    name = os.path.basename(str(r.path)) or str(r.path)
    summary = {
        "tool": "basira", "version": r.version, "file": name, "sha256": sha256,
        "size": file_size, "media_type": r.media_type, "verdict": r.verdict,
        "verdict_key": _verdict_key(r), "verdict_level": int(r.verdict_level),
        "scores": {k: round(float(v), 4) for k, v in r.scores.items()},
        "generated": generated.isoformat(timespec="seconds"), "lang": ctx.lang,
    }
    nav = "".join(f'<li><a href="#{a}">{_esc(label)}</a></li>' for a, label in toc)
    direction = "rtl" if ctx.rtl else "ltr"
    return (
        f'<!doctype html>\n<html lang="{ctx.lang}" dir="{direction}" '
        f'data-close="{_esc(t("close"))}" data-enlarge="{_esc(t("enlarge"))}"><head>'
        f'<meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="referrer" content="no-referrer">'
        f'<meta name="generator" content="Basira {_esc(r.version)}">'
        f'<title>{_esc(t("title"))} — {_esc(name)}</title><style>{CSS}</style></head>'
        f'<body><a class="skip" href="#verdict">{_esc(t("skip"))}</a><div class="wrap">'
        f'<header class="mast"><div><div class="brand">Basira <span>· بصيرة</span></div>'
        f'<h1>{_esc(t("report"))}</h1></div>'
        f'<button type="button" class="btn js-only" data-print>{_esc(t("print"))}</button>'
        f'</header>'
        f'<nav class="toc" aria-label="{_esc(t("nav"))}"><ul>{nav}</ul></nav>'
        f'<main id="main">{_facts(ctx, r, sha256, file_size, generated)}{body}</main>'
        f'<footer class="foot"><h2>{_esc(t("howto"))}</h2><p>{_esc(t("disclaimer"))}</p>'
        f'<p>{_esc(t("offline"))} {_ltr(f"Basira {r.version}".strip())}</p></footer></div>'
        f'<script type="application/json" id="basira-summary">{_json_for_script(summary)}'
        f'</script><script>{JS}</script></body></html>')


def save_html(report: Report, path: str, lang: str = "en", **kwargs: Any) -> str:
    """Write :func:`render_html` to ``path`` (UTF-8) and return ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(report, lang, **kwargs))
    return path
