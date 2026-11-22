// Builds presentation/EchoGesture_Presentation.pptx from the measured project reports.
// Usage: NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node build_presentation.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(__dirname, "EchoGesture_Presentation.pptx");
const APPLY_THEME = process.env.PPTX_APPLY_THEME;
const read = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));

const evalReport = read("results/reports/evaluation_report.json");
const exportReport = read("results/reports/export_report.json");
const live = read("results/reports/live_benchmark.json");
const deployed = read("results/reports/deployed_eval.json");
const bench = read("results/reports/benchmark_report.json");
const training = read("results/reports/training_report_tcnn.json");
const dataset = read("results/reports/dataset_report.json");
const kpv = read("results/reports/keypoint_validation_final.json");
const analysis = read("results/reports/test_analysis.json");
const calibration = read("results/reports/calibration_analysis.json");
const history = fs
  .readFileSync(path.join(ROOT, "models/checkpoints/tcnn/history.csv"), "utf8")
  .trim()
  .split("\n");
const header = history.shift().split(",");
const col = (name) => history.map((line) => Number(line.split(",")[header.indexOf(name)]));

const pct = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const ms = (v, d = 1) => `${v.toFixed(d)} ms`;
const test = exportReport.test;
const stream = live.prediction_latency_ms;

const THEME = {
  name: "EchoGesture",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14212B",
    lt1: "FFFFFF",
    dk2: "0B3C49",
    lt2: "EAF2F3",
    accent1: "0B7A75",
    accent2: "E8A33D",
    accent3: "C8463D",
    accent4: "3A6EA5",
    accent5: "6B7C88",
    accent6: "2E8B57",
    hlink: "0B7A75",
    folHlink: "5E4B8B",
  },
};
const HEX = THEME.colors;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: `#${color}`, size: String(size) })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "EchoGesture: on-device isolated ASL word recognition";
  pres.subject = "Samsung PRISM submission";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "Dark",
    background: { color: C.text2 },
    objects: [
      {
        placeholder: {
          options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.4, fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left" },
          text: "",
        },
      },
      {
        placeholder: {
          options: { name: "body", type: "body", x: 0.8, y: 3.75, w: 11.7, h: 1.2, fontSize: 20, color: "CFE3E5", valign: "top", align: "left" },
          text: "",
        },
      },
    ],
  });
  pres.defineSlideMaster({
    title: "Content",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      {
        placeholder: {
          options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 34, bold: true, color: C.text2, valign: "middle", align: "left" },
          text: "",
        },
      },
      { text: { text: "EchoGesture · Samsung PRISM submission", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent5 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });

  const icons = {
    hand: await icon(fa.FaSignLanguage, HEX.accent1),
    lock: await icon(fa.FaLock, HEX.accent1),
    bolt: await icon(fa.FaBolt, HEX.accent1),
    video: await icon(fa.FaVideo, HEX.accent1),
    chip: await icon(fa.FaMicrochip, HEX.accent1),
    db: await icon(fa.FaDatabase, HEX.accent1),
    warn: await icon(fa.FaExclamationTriangle, HEX.accent2),
    check: await icon(fa.FaCheckCircle, HEX.accent6),
    cross: await icon(fa.FaTimesCircle, HEX.accent3),
    road: await icon(fa.FaRoute, HEX.accent1),
    users: await icon(fa.FaUserFriends, HEX.accent1),
    eye: await icon(fa.FaEye, HEX.accent1),
    handW: await icon(fa.FaSignLanguage, "FFFFFF"),
  };

  let n = 0;
  const content = (section, title) => {
    const s = pres.addSlide({ masterName: "Content", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    n += 1;
    return s;
  };
  const card = (s, name, x, y, w, h, fill = C.background2) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: name, x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
  const iconDot = (s, name, data, x, y, d = 0.62) => {
    s.addShape(pres.shapes.OVAL, { objectName: `${name} circle`, x, y, w: d, h: d, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.5 } });
    s.addImage({ data, x: x + d * 0.2, y: y + d * 0.2, w: d * 0.6, h: d * 0.6, altText: name });
  };
  const text = (s, value, opts) => s.addText(value, { isTextBox: true, margin: 0, ...opts });
  const bullets = (items, fontSize = 15) =>
    items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize, color: C.text1, paraSpaceAfter: 6 } }));
  const arrow = (s, name, x1, y1, x2, y2) =>
    s.addShape(pres.shapes.LINE, {
      objectName: name,
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
      flipH: x2 < x1, flipV: y2 < y1,
      line: { color: C.accent5, width: 1.75, endArrowType: "triangle" },
    });
  const stat = (s, name, x, y, w, value, label, color = C.accent1) => {
    text(s, value, { objectName: `${name} value`, x, y, w, h: 0.8, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace });
    text(s, label, { objectName: `${name} label`, x, y: y + 0.8, w, h: 0.6, fontSize: 12, color: C.accent5 });
  };

  // 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "Dark", sectionTitle: "Introduction" });
  n += 1;
  s.addImage({ data: icons.handW, x: 0.8, y: 1.0, w: 0.9, h: 0.9, altText: "Sign language icon" });
  s.addText("EchoGesture", { placeholder: "title" });
  s.addText("On-device isolated ASL word recognition from body and hand landmarks", { placeholder: "body" });
  text(s, "Samsung PRISM submission  ·  github.com/shr1989/SAMSUNG_PRISM", { objectName: "repo", x: 0.8, y: 6.2, w: 11.7, h: 0.4, fontSize: 14, color: "CFE3E5" });
  s.addNotes("Title. Team member names are not listed in the project files; add them here before presenting.");

  // 2. Problem statement
  s = content("Introduction", "Problem statement");
  text(s, "Recognise which of 2,000 ASL words is signed in a short video clip, entirely on the device.", { objectName: "lede", x: 0.6, y: 1.35, w: 12.1, h: 0.6, fontSize: 18, color: C.text1 });
  const rows = [
    ["Constraint", "Target"],
    ["Model size (quantized)", "≤ 20 MB"],
    ["Latency (p95)", "≤ 50 ms"],
    ["Top-1 accuracy on held-out clips", "≥ 70%"],
    ["Accuracy drop with an occluded hand", "≤ 5 points"],
    ["Average power", "≤ 500 mW"],
    ["Latency drift after 10 minutes", "≤ 20%"],
    ["Cloud processing", "None"],
  ];
  s.addTable(
    rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.text2 : i % 2 ? C.background2 : C.background1 } } }))),
    { objectName: "constraints table", x: 0.6, y: 2.15, w: 7.2, colW: [4.8, 2.4], fontSize: 14, rowH: 0.48, border: { type: "none" } }
  );
  card(s, "input card", 8.4, 2.15, 4.3, 3.84);
  iconDot(s, "video", icons.video, 8.7, 2.4);
  text(s, "Input", { objectName: "input head", x: 9.5, y: 2.45, w: 3, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
  text(s, "A third-person camera clip of one isolated sign", { objectName: "input body", x: 8.7, y: 3.15, w: 3.8, h: 0.8, fontSize: 14, color: C.text1 });
  text(s, "Output", { objectName: "output head", x: 9.5, y: 4.1, w: 3, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
  iconDot(s, "output", icons.hand, 8.7, 4.05);
  text(s, "The word as a caption, with confidence and top-5 alternatives", { objectName: "output body", x: 8.7, y: 4.8, w: 3.8, h: 0.9, fontSize: 14, color: C.text1 });
  text(s, "Targets from the EchoGesture track solution guide.", { objectName: "source", x: 0.6, y: 6.2, w: 7.2, h: 0.3, fontSize: 11, color: C.accent5 });
  s.addNotes("The task is isolated word classification: one sign per clip, one label out. The table lists the hard limits from the track guide.");

  // 3. Motivation
  s = content("Introduction", "Why on-device sign recognition matters");
  const motives = [
    [icons.users, "Communication access", "Signers and non-signers need a way to understand each other in everyday conversation."],
    [icons.lock, "Privacy", "The camera also films bystanders, so no video may leave the phone. Every step must run locally."],
    [icons.bolt, "Conversation pace", "Captions that lag break the flow of a conversation, so latency is a first-class requirement."],
  ];
  motives.forEach(([ic, head, body], i) => {
    const x = 0.6 + i * 4.15;
    card(s, `motive ${i + 1}`, x, 1.6, 3.85, 3.9);
    iconDot(s, head, ic, x + 0.35, 1.9, 0.8);
    text(s, head, { objectName: `${head} head`, x: x + 0.35, y: 2.95, w: 3.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
    text(s, body, { objectName: `${head} body`, x: x + 0.35, y: 3.55, w: 3.2, h: 1.7, fontSize: 15, color: C.text1, valign: "top" });
  });
  s.addNotes("Motivation, without statistics we cannot source: access, privacy and latency drive the design.");

  // 4. Proposed solution
  pres.addSection({ title: "Approach" });
  s = content("Approach", "Proposed solution: landmarks, not pixels");
  const steps = ["Video frames", "MediaPipe Holistic\n75 landmarks", "Normalise +\n64-frame resample", "576 features\nper frame", "Temporal 1D-CNN\n(ONNX)", "Word + top-5"];
  steps.forEach((label, i) => {
    const x = 0.6 + i * 2.07;
    card(s, `step ${i + 1}`, x, 1.7, 1.8, 1.3, i === 4 ? C.accent1 : C.background2);
    text(s, label, { objectName: `step ${i + 1} text`, x: x + 0.1, y: 1.75, w: 1.6, h: 1.2, fontSize: 14, bold: true, align: "center", valign: "middle", color: i === 4 ? C.background1 : C.text2 });
    if (i < steps.length - 1) arrow(s, `arrow ${i + 1}`, x + 1.82, 2.35, x + 2.05, 2.35);
  });
  s.addText(
    bullets([
      "A frame is ~200,000 numbers; the sign lives in where the arms, hands and fingers move.",
      "MediaPipe turns each frame into 75 landmarks, so the classifier stays tiny, fast and less sensitive to lighting and background.",
      "One preprocessing function is shared by training, evaluation and live inference.",
      "The final model is a pure temporal 1D-CNN: no recurrent or attention layers, which keeps it simple to quantize.",
    ]),
    { objectName: "solution bullets", x: 0.6, y: 3.3, w: 12.1, h: 1.9, valign: "top" }
  );
  stat(s, "landmark numbers", 0.6, 5.35, 3.8, "225", "numbers per frame from 75 landmarks × (x, y, z)");
  stat(s, "model params", 4.6, 5.35, 3.8, "0.87 M", "classifier parameters");
  stat(s, "model latency", 8.6, 5.35, 4.1, `${exportReport.latency_batch1_ms.onnx_fp32.p95.toFixed(2)} ms`, "model p95 on one CPU thread (excludes MediaPipe)");
  s.addNotes("Pipeline overview. Stress that the same preprocess() runs at train and test time.");

  // 5. Dataset
  s = content("Approach", "Dataset: WLASL-2000");
  stat(s, "videos", 0.6, 1.5, 2.9, dataset.annotated_instances.toLocaleString("en-US"), "videos in WLASL-2000 (all decodable)");
  stat(s, "classes", 3.6, 1.5, 2.9, "2,000", "ASL words (glosses)");
  stat(s, "signers", 6.6, 1.5, 2.9, String(dataset.unique_signers), "signers");
  stat(s, "used", 9.6, 1.5, 3.1, "10,000", "videos used: 3 train / 1 val / 1 test per word", C.accent2);
  s.addText(
    bullets([
      `Official NSLT-2000 split: ${dataset.split_counts.train.toLocaleString("en-US")} train / ${dataset.split_counts.val.toLocaleString("en-US")} val / ${dataset.split_counts.test.toLocaleString("en-US")} test videos.`,
      "A deterministic 10,000-video subset keeps all 2,000 words while fitting local extraction and training time on a fanless MacBook Air M4.",
      "The test set was evaluated once and never used for tuning or calibration.",
      `Results are signer-dependent: ${analysis.signer_breakdown.clips_by_seen_signers.toLocaleString("en-US")} of 2,000 test clips come from signers also seen in training.`,
    ]),
    { objectName: "dataset bullets", x: 0.6, y: 3.15, w: 12.1, h: 2.9, valign: "top" }
  );
  text(s, "Li et al., \"Word-level Deep Sign Language Recognition from Video\" (WACV 2020). WLASL is licensed for academic, computational use (C-UDA).", { objectName: "citation", x: 0.6, y: 6.25, w: 12.1, h: 0.5, fontSize: 11, color: C.accent5 });
  s.addNotes("Dataset numbers come from results/reports/dataset_report.json and final_selection_report.json.");

  // 6. Dataset preprocessing
  s = content("Approach", "Dataset preprocessing");
  card(s, "prep card", 0.6, 1.5, 5.6, 4.9);
  iconDot(s, "dataset", icons.db, 0.9, 1.75);
  text(s, "Preparing the data", { objectName: "prep head", x: 1.7, y: 1.8, w: 4.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  s.addText(
    bullets([
      "Official NSLT-2000 splits are authoritative; all 21,095 videos were checked and decode.",
      "Deterministic subset: 3 train / 1 val / 1 test clip for every one of the 2,000 words.",
      "Clips are cut to their annotated sign segment where WLASL gives one.",
      "Feature statistics come from the 6,000 training clips only; the test set was evaluated once.",
    ], 14),
    { objectName: "prep bullets", x: 0.9, y: 2.55, w: 5.1, h: 3.7, valign: "top" }
  );
  const pre = [
    ["1", "Fill short gaps", "Interpolate interior gaps up to 4 samples; hold trailing gaps."],
    ["2", "Normalise", "Centre on the clip-median shoulder midpoint, scale by shoulder width, drop legs."],
    ["3", "Fix length", "Clip to ±6 shoulder widths, resample linearly to 64 frames."],
    ["4", "Mark missing", "Undetected landmarks become exactly 0."],
  ];
  pre.forEach(([num, head, body], i) => {
    const y = 1.5 + i * 1.25;
    s.addShape(pres.shapes.OVAL, { objectName: `pre ${num} circle`, x: 6.6, y, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    text(s, num, { objectName: `pre ${num} num`, x: 6.6, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, head, { objectName: `pre ${num} head`, x: 7.35, y: y - 0.05, w: 5.3, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    text(s, body, { objectName: `pre ${num} body`, x: 7.35, y: y + 0.38, w: 5.3, h: 0.7, fontSize: 14, color: C.text1, valign: "top" });
  });
  s.addNotes("All four steps live in src/preprocess.py::preprocess, the single transform used by training, the app, main.py and the HUD.");

  // 7. MediaPipe landmark extraction
  s = content("Approach", "MediaPipe landmark extraction");
  [["33", "pose landmarks", C.accent1], ["21", "left-hand landmarks", C.accent2], ["21", "right-hand landmarks", C.accent4]].forEach(([num, label, color], i) => {
    const x = 0.6 + i * 4.15;
    card(s, `landmark group ${i + 1}`, x, 1.5, 3.85, 1.8);
    text(s, num, { objectName: `landmark count ${i + 1}`, x: x + 0.3, y: 1.6, w: 3.2, h: 0.9, fontSize: 48, bold: true, color, fontFace: THEME.headFontFace });
    text(s, `${label}, each (x, y, z)`, { objectName: `landmark label ${i + 1}`, x: x + 0.3, y: 2.55, w: 3.3, h: 0.5, fontSize: 15, color: C.text1 });
  });
  s.addText(
    bullets([
      "MediaPipe Holistic 0.10.14 (pretrained, not trained by us) runs on every 2nd frame: 12.5 samples per second at 25 fps.",
      "Each sampled frame becomes a (75, 3) array; a group that is not detected is stored as NaN and handled in preprocessing.",
      `${kpv.counts.completed.toLocaleString("en-US")} of 10,000 videos extracted, 0 failed or corrupted; ${kpv.frames_in_completed_arrays.toLocaleString("en-US")} sampled frames.`,
      `${pct(kpv.missing_hand_slots_fraction, 0)} of hand slots are undetected (often a resting or out-of-frame hand); left/right ordering was verified against the pose wrists.`,
      "Hand identity is anatomical on unmirrored video, so live input must not be mirrored.",
    ], 15),
    { objectName: "extraction bullets", x: 0.6, y: 3.6, w: 12.1, h: 2.9, valign: "top" }
  );
  s.addNotes("Numbers from results/reports/keypoint_validation_final.json.");

  // 8. Feature engineering
  s = content("Approach", "Feature engineering: 576 numbers per frame");
  const feats = [
    ["225", "Body + hand positions", "75 landmarks × xyz after normalisation", C.accent1],
    ["126", "Upright hand shape", "Each hand relative to its wrist, rotated so wrist → middle knuckle points up, unit length", C.accent4],
    ["225", "Velocity", "Frame-to-frame motion, masked where a landmark is missing", C.accent2],
  ];
  feats.forEach(([num, head, body, color], i) => {
    const x = 0.6 + i * 4.15;
    card(s, `feature ${i + 1}`, x, 1.55, 3.85, 3.1);
    text(s, num, { objectName: `feature ${i + 1} count`, x: x + 0.3, y: 1.75, w: 3.2, h: 0.9, fontSize: 48, bold: true, color, fontFace: THEME.headFontFace });
    text(s, head, { objectName: `feature ${i + 1} head`, x: x + 0.3, y: 2.7, w: 3.3, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
    text(s, body, { objectName: `feature ${i + 1} body`, x: x + 0.3, y: 3.25, w: 3.3, h: 1.3, fontSize: 14, color: C.text1, valign: "top" });
  });
  s.addText(
    bullets([
      "The upright hand-shape features give the same numbers for the same handshape whether the hand is high, low, near or tilted.",
      "Features and standardisation (statistics fitted on the 6,000 training clips only) are computed inside the model graph, so the ONNX file takes raw (64, 75, 3) landmarks.",
    ]),
    { objectName: "feature bullets", x: 0.6, y: 4.95, w: 12.1, h: 1.6, valign: "top" }
  );
  s.addNotes("225 + 126 + 225 = 576. In-graph features remove any chance of a train/serve mismatch.");

  // 9. Architecture
  s = content("Approach", "Model: a pure temporal 1D-CNN");
  const arch = [
    ["Input", "64 × 75 × 3 landmarks"],
    ["In-graph features", "64 × 576, standardised"],
    ["Linear + LayerNorm + ReLU", "576 → 192"],
    ["Conv block (k = 5)", "BN, ReLU, residual · 64 steps"],
    ["Residual down block (k = 5)", "stride 2 · 32 steps"],
    ["Temporal mean pool", "192"],
    ["Dropout 0.3 + Linear", "192 → 2,000"],
    ["Softmax", "word + top-5"],
  ];
  arch.forEach(([head, sub], i) => {
    const row = Math.floor(i / 4);
    const colI = row === 0 ? i : 7 - i;
    const x = 0.6 + colI * 3.1;
    const y = 1.6 + row * 1.9;
    const hl = i >= 2 && i <= 6;
    card(s, `layer ${i + 1}`, x, y, 2.7, 1.3, hl ? C.accent1 : C.background2);
    text(s, head, { objectName: `layer ${i + 1} head`, x: x + 0.15, y: y + 0.15, w: 2.4, h: 0.5, fontSize: 15, bold: true, color: hl ? C.background1 : C.text2, align: "center" });
    text(s, sub, { objectName: `layer ${i + 1} sub`, x: x + 0.15, y: y + 0.65, w: 2.4, h: 0.55, fontSize: 12, color: hl ? C.background1 : C.text1, align: "center" });
    if (i < 3) arrow(s, `arch arrow ${i + 1}`, x + 2.72, y + 0.65, x + 3.08, y + 0.65);
    if (i > 3 && i < 7) arrow(s, `arch arrow ${i + 1}`, x - 0.02, y + 0.65, x - 0.38, y + 0.65);
  });
  arrow(s, "arch arrow down", 0.6 + 3 * 3.1 + 1.35, 2.92, 0.6 + 3 * 3.1 + 1.35, 3.48);
  stat(s, "params", 0.6, 5.35, 3.6, training.parameters.toLocaleString("en-US"), "parameters");
  stat(s, "fp32", 4.6, 5.35, 3.6, `${exportReport.size_mb.onnx_fp32.toFixed(2)} MB`, "FP32 ONNX (deployed)");
  stat(s, "int8", 8.6, 5.35, 3.6, `${exportReport.size_mb.onnx_int8.toFixed(2)} MB`, "INT8 ONNX (limit 20 MB)");
  s.addNotes("No GRU, LSTM, transformer or 3D-CNN. Convolutions over time only; everything is parallel and quantizes cleanly.");

  // 10. Training
  s = content("Approach", "Training methodology");
  s.addImage({ path: path.join(__dirname, "assets/training_curve.png"), x: 0.6, y: 1.45, w: 7.4, h: 4.9, altText: "Validation top-1 and top-5 accuracy by epoch; epoch 95 selected" });
  s.addText(
    bullets([
      "AdamW (lr 0.002, weight decay 0.05), one-cycle schedule, label smoothing 0.1, batch 64, 150 epochs.",
      `Checkpoint chosen by validation macro-F1: epoch ${training.best_epoch} (val top-1 ${pct(evalReport.val.clean.top1)}).`,
      "On-device augmentation, training only: time crop, mirror with hand swap, ±20° yaw / ±10° roll, scale, shift, jitter, hand hiding, point dropout.",
      `Apple M4 GPU (MPS): ${training.mean_epoch_seconds.toFixed(1)} s per epoch, ${training.training_minutes.toFixed(1)} min in total.`,
    ], 14),
    { objectName: "training bullets", x: 8.3, y: 1.5, w: 4.4, h: 4.9, valign: "top" }
  );
  s.addNotes("Curve from models/checkpoints/tcnn/history.csv. Test data was never loaded during training or selection.");

  // 11. Export and quantization
  pres.addSection({ title: "Deployment" });
  s = content("Deployment", "ONNX export and INT8 quantization");
  const q = [
    ["", "PyTorch FP32", "ONNX FP32", "ONNX INT8"],
    ["Test top-1", pct(test.torch_fp32.top1), pct(test.onnx_fp32.top1), pct(test.onnx_int8.top1)],
    ["Test top-5", pct(test.torch_fp32.top5), pct(test.onnx_fp32.top5), pct(test.onnx_int8.top5)],
    ["File size", "-", `${exportReport.size_mb.onnx_fp32.toFixed(2)} MB`, `${exportReport.size_mb.onnx_int8.toFixed(2)} MB`],
    ["Model p95 latency", "-", ms(exportReport.latency_batch1_ms.onnx_fp32.p95, 2), ms(exportReport.latency_batch1_ms.onnx_int8.p95, 2)],
  ];
  s.addTable(
    q.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.text2 : i % 2 ? C.background2 : C.background1 }, align: j ? "center" : "left" } }))),
    { objectName: "quantization table", x: 0.6, y: 1.5, w: 7.6, colW: [2.2, 1.8, 1.8, 1.8], fontSize: 14, rowH: 0.55, border: { type: "none" } }
  );
  s.addText(
    bullets([
      `ONNX FP32 matches PyTorch: max logit difference ${test.onnx_fp32_vs_torch.max_abs_logit_diff.toExponential(1)}, 100% top-1 agreement.`,
      "Static QDQ INT8 calibrated on 300 training clips (never test). Conv, MatMul and Gemm run in INT8; feature ops, LayerNorm and pooling stay FP32.",
      `INT8 costs ${(100 * (test.onnx_fp32.top1 - test.onnx_int8.top1)).toFixed(2)} points of top-1, but agrees with FP32 on only ${pct(test.onnx_int8_vs_fp32.argmax_agreement, 1)} of test clips, so FP32 is the deployed model.`,
    ], 14),
    { objectName: "quantization bullets", x: 8.5, y: 1.5, w: 4.2, h: 4.9, valign: "top" }
  );
  text(s, "Latency: 20 warm-up + 200 timed runs, batch 1, ONNX Runtime 1.32 CPU, 1 thread, Apple M4.", { objectName: "latency note", x: 0.6, y: 4.5, w: 7.6, h: 0.5, fontSize: 11, color: C.accent5 });
  s.addNotes("From results/reports/export_report.json.");

  // 12. Application architecture
  s = content("Deployment", "Application architecture");
  const sys = ["Browser:\nupload or\nwebcam", "Gradio app.py\ninput checks", "MediaPipe\nHolistic", "preprocess()\n64 × 75 × 3", "ONNX Runtime\nFP32 1D-CNN", "Top-5 words +\ntiming"];
  sys.forEach((label, i) => {
    const x = 0.6 + i * 2.07;
    const hl = i === 4;
    card(s, `system ${i + 1}`, x, 1.55, 1.8, 1.3, hl ? C.accent1 : C.background2);
    text(s, label, { objectName: `system ${i + 1} text`, x: x + 0.05, y: 1.6, w: 1.7, h: 1.2, fontSize: 14, bold: true, align: "center", valign: "middle", color: hl ? C.background1 : C.text2 });
    if (i < sys.length - 1) arrow(s, `system arrow ${i + 1}`, x + 1.82, 2.2, x + 2.05, 2.2);
  });
  const entry = [
    ["Web app: app.py", ["Gradio UI for upload and webcam recording.", "Validates type, size, length, frames and landmarks.", "Loads the model once; one request at a time."]],
    ["Offline API: main.py", ["predict(video_path) returns JSON: word, confidence, top-5, latency.", "Produced predictions.jsonl for all 2,000 test clips."]],
    ["Live HUD: hud_demo.py", ["Segments signs by hand presence and captions them.", "Source of the live latency, drift and power measurements."]],
  ];
  entry.forEach(([head, items], i) => {
    const x = 0.6 + i * 4.15;
    card(s, `entry ${i + 1}`, x, 3.2, 3.85, 2.75);
    text(s, head, { objectName: `entry ${i + 1} head`, x: x + 0.25, y: 3.35, w: 3.4, h: 0.5, fontSize: 17, bold: true, color: C.text2 });
    s.addText(bullets(items, 13), { objectName: `entry ${i + 1} bullets`, x: x + 0.25, y: 3.9, w: 3.45, h: 1.95, valign: "top" });
  });
  text(s, "All three share src/inference.py (Recognizer), src/extract_keypoints.py and src/preprocess.py, so every path runs the identical pipeline. No cloud API is called.", { objectName: "shared note", x: 0.6, y: 6.15, w: 12.1, h: 0.5, fontSize: 13, color: C.accent5 });
  s.addNotes("The web app reuses the existing modules; nothing in the pipeline was reimplemented.");

  // 13. Interactive demo application
  s = content("Deployment", "Interactive demo application");
  s.addImage({ path: path.join(ROOT, "docs/images/app_screenshot.png"), x: 0.6, y: 1.45, w: 7.4, h: 4.89, altText: "Screenshot of the EchoGesture Gradio app showing a prediction, top-5 scores and timing" });
  text(s, "Actual run on WLASL test clip 60522 (true word UNDER): the model predicted CRUEL, a miss. Video preview hidden for this codec.", { objectName: "screenshot caption", x: 0.6, y: 6.38, w: 7.4, h: 0.45, fontSize: 11, color: C.accent5 });
  card(s, "verified card", 8.3, 1.45, 4.4, 2.75);
  text(s, "Verified", { objectName: "verified head", x: 8.55, y: 1.55, w: 3.9, h: 0.45, fontSize: 17, bold: true, color: C.accent6 });
  s.addText(bullets(["Upload and prediction in the browser UI.", "Same top-5 and scores as main.predict(): 12/12 direct, 4/4 over HTTP, 3/3 from a clean Space build.", "Webcam recording in the browser (two real recordings, Windows, Brave) and clear errors for bad videos."], 12), { objectName: "verified bullets", x: 8.55, y: 2.0, w: 4.0, h: 2.15, valign: "top" });
  card(s, "pending card", 8.3, 4.35, 4.4, 2.0, "FDF3E1");
  text(s, "Not yet verified", { objectName: "pending head", x: 8.55, y: 4.45, w: 3.9, h: 0.45, fontSize: 17, bold: true, color: C.accent2 });
  s.addText(bullets(["Live caption HUD (hud_demo.py) with a real camera.", "Public hosting: Hugging Face Space files are prepared, not deployed."], 12), { objectName: "pending bullets", x: 8.55, y: 4.9, w: 4.0, h: 1.4, valign: "top" });
  s.addNotes("Launch locally with: python app.py. The screenshot is a genuine run, including a wrong prediction.");

  // 14. Evaluation results
  pres.addSection({ title: "Results" });
  s = content("Results", "Evaluation results on the held-out test set");
  stat(s, "top1", 0.6, 1.4, 3.0, pct(evalReport.test.clean.top1), "test top-1 (95% CI 17.25-20.60%)", C.accent3);
  stat(s, "top5", 3.7, 1.4, 3.0, pct(evalReport.test.clean.top5), "test top-5");
  stat(s, "f1", 6.8, 1.4, 2.6, evalReport.test.clean.macro_f1.toFixed(4), "macro F1");
  stat(s, "target", 9.6, 1.4, 3.1, "≥ 70%", "top-1 target: not met", C.accent3);
  s.addImage({ path: path.join(__dirname, "assets/calibration.png"), x: 0.6, y: 3.05, w: 6.0, h: 3.6, altText: "Reliability chart: accuracy per score bin is below the diagonal" });
  s.addText(
    bullets([
      "2,000 test clips, 1 per word, official split, evaluated once with the FP32 model.",
      `Signer overlap: ${analysis.signer_breakdown.clips_by_seen_signers.toLocaleString("en-US")} of 2,000 test clips come from signers seen in training. Same signer and same word in training: ${pct(analysis.signer_breakdown.accuracy_same_signer_same_word, 1)}; others ${pct(analysis.signer_breakdown.accuracy_other, 1)}. Unseen-signer accuracy is not established.`,
      `Confidence is not calibrated: mean top-1 score ${pct(calibration.mean_top1_confidence, 1)} vs ${pct(calibration.top1_accuracy, 1)} accuracy.`,
      `${evalReport.test_per_class_summary.classes_with_any_correct} of 2,000 words were recognised at least once.`,
    ], 13),
    { objectName: "results bullets", x: 6.9, y: 3.1, w: 5.8, h: 3.55, valign: "top" }
  );
  s.addNotes("Headline: 18.90% top-1, 42.60% top-5 on 2,000 classes, far below the 70% target. Calibration figures from results/reports/calibration_analysis.json.");

  // 15. Latency and size
  s = content("Results", "Latency and model-size benchmarks");
  s.addImage({ path: path.join(__dirname, "assets/latency.png"), x: 0.6, y: 1.45, w: 7.4, h: 3.6, altText: "Measured p95 latency: model, preprocess, MediaPipe per frame, sign end to word" });
  card(s, "delay card", 8.3, 1.45, 4.4, 1.7, "FDF3E1");
  text(s, `+ ${live.decision_delay_ms.toFixed(0)} ms`, { objectName: "delay value", x: 8.55, y: 1.55, w: 4, h: 0.7, fontSize: 32, bold: true, color: C.accent2, fontFace: THEME.headFontFace });
  text(s, "Deliberate end-of-sign confirmation wait (4 samples without hands). Not processing time; excluded from the 32.2 ms.", { objectName: "delay text", x: 8.55, y: 2.25, w: 4, h: 0.85, fontSize: 12, color: C.text1 });
  card(s, "file card", 8.3, 3.35, 4.4, 1.7);
  text(s, ms(bench.latency_ms.onnx_fp32_end_to_end_ms.p95, 0), { objectName: "file value", x: 8.55, y: 3.45, w: 4, h: 0.7, fontSize: 32, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  text(s, "p95 for a whole recorded video file through predict(): MediaPipe on every sampled frame dominates.", { objectName: "file text", x: 8.55, y: 4.15, w: 4, h: 0.85, fontSize: 12, color: C.text1 });
  stat(s, "drift", 0.6, 5.3, 3.8, `+${live.landmarks_drift.p95_drift_percent.toFixed(1)}%`, "p95 latency drift over 10.5 minutes (target ≤ 20%)", C.accent6);
  stat(s, "size", 4.6, 5.3, 3.6, `${exportReport.size_mb.onnx_int8.toFixed(2)} MB`, `INT8 model; FP32 ${exportReport.size_mb.onnx_fp32.toFixed(2)} MB (target ≤ 20 MB)`, C.accent6);
  stat(s, "power", 8.4, 5.3, 4.3, `${live.summary.average_power_w.toFixed(2)} W`, "active CPU power estimate, per-process (target 500 mW: not met)", C.accent3);
  s.addNotes("Live latency from a 10.5-minute real-time replay (213 signs). Power is the macOS per-process CPU energy counter: it excludes camera, display and memory, so total device power is higher. Not a phone measurement.");

  // 16. Robustness
  s = content("Results", "Robustness evaluation");
  s.addImage({ path: path.join(__dirname, "assets/test_accuracy.png"), x: 0.6, y: 1.45, w: 7.2, h: 3.55, altText: "Test top-1 and top-5 accuracy for clean, occluded and rotated inputs" });
  const robust = [
    ["One hand removed from keypoints, 30% of frames (2,000 clips)", `−${evalReport.test.occluded_top1_drop_pp.toFixed(1)} pts top-1`],
    ["Keypoints rotated (2,000 clips)", `−${evalReport.test.rotated_top1_drop_pp.toFixed(1)} pts top-1`],
    ["Dominant hand blacked out in the video pixels, 30% of frames (400 clips)", `−${deployed.occlusion_top1_drop_pp.toFixed(2)} pts top-1 (95% CI 1.5-8.0); −${deployed.occlusion_top5_drop_pp.toFixed(1)} pts top-5`],
    ["Hand detection inside the blacked-out window", `${pct(deployed.clean_hand_detected_in_window, 0)} → ${pct(deployed.occluded_hand_detected_in_window, 0)}`],
    ["Live streaming replay, 213 signs", `top-1 ${pct(live.streaming_top1, 1)}, top-5 ${pct(live.streaming_top5, 1)}`],
  ];
  robust.forEach(([what, value], i) => {
    const y = 1.45 + i * 1.02;
    card(s, `robust ${i + 1}`, 8.1, y, 4.6, 0.9, i % 2 ? C.background1 : C.background2);
    text(s, what, { objectName: `robust ${i + 1} what`, x: 8.3, y: y + 0.06, w: 4.25, h: 0.45, fontSize: 12, color: C.text1, valign: "top" });
    text(s, value, { objectName: `robust ${i + 1} value`, x: 8.3, y: y + 0.5, w: 4.25, h: 0.35, fontSize: 13, bold: true, color: C.text2 });
  });
  text(s, "The ≤ 5-point occlusion target holds on point estimates; the pixel-level interval reaches 8 points, so it is not statistically established. Angled cameras, dim light and new signers are untested.", { objectName: "robust note", x: 0.6, y: 5.25, w: 7.2, h: 1.0, fontSize: 13, color: C.text1, valign: "top" });
  s.addNotes("Sources: evaluation_report.json, deployed_eval.json, live_benchmark.json.");

  // 17. Scorecard
  s = content("Results", "Requirements scorecard");
  const score = [
    ["Model size ≤ 20 MB", `${exportReport.size_mb.onnx_int8.toFixed(2)} MB INT8 / ${exportReport.size_mb.onnx_fp32.toFixed(2)} MB FP32`, true],
    ["Latency p95 ≤ 50 ms", `${stream.end_to_end_ms.p95.toFixed(1)} ms sign end → word (excludes the 320 ms wait)`, true],
    ["Top-1 ≥ 70%", `${pct(evalReport.test.clean.top1)} on 2,000 test clips`, false],
    ["Occlusion drop ≤ 5 points", `${evalReport.test.occluded_top1_drop_pp.toFixed(1)} (keypoints) / ${deployed.occlusion_top1_drop_pp.toFixed(2)} (pixels, CI to 8.0)`, true],
    ["Power ≤ 500 mW", `${live.summary.average_power_w.toFixed(2)} W per-process CPU estimate (laptop)`, false],
    ["Thermal drift ≤ 20%", `+${live.landmarks_drift.p95_drift_percent.toFixed(1)}% over 10.5 minutes`, true],
    ["No cloud", "All processing local", true],
  ];
  score.forEach(([req, measured, ok], i) => {
    const y = 1.45 + i * 0.75;
    card(s, `score row ${i + 1}`, 0.6, y, 12.1, 0.62, i % 2 ? C.background1 : C.background2);
    s.addImage({ data: ok ? icons.check : icons.cross, x: 0.8, y: y + 0.11, w: 0.4, h: 0.4, altText: ok ? "pass" : "fail" });
    text(s, req, { objectName: `score ${i + 1} req`, x: 1.45, y: y + 0.1, w: 4.0, h: 0.42, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    text(s, measured, { objectName: `score ${i + 1} measured`, x: 5.5, y: y + 0.1, w: 5.6, h: 0.42, fontSize: 14, color: C.text1, valign: "middle" });
    text(s, ok ? "Pass" : "Fail", { objectName: `score ${i + 1} status`, x: 11.2, y: y + 0.1, w: 1.3, h: 0.42, fontSize: 15, bold: true, color: ok ? C.accent6 : C.accent3, valign: "middle", align: "right" });
  });
  text(s, "Occlusion passes on the point estimate only. All values measured on a MacBook Air M4, not on a phone.", { objectName: "score note", x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 11, color: C.accent5 });
  s.addNotes("Five of seven targets met; accuracy and power are not.");

  // 18. Limitations
  pres.addSection({ title: "Outlook" });
  s = content("Outlook", "Limitations and challenges");
  const limits = [
    ["Accuracy", "18.90% top-1 on 2,000 words with 3 training clips per word, far below the 70% target."],
    ["Generalisation", "1,997 of 2,000 test clips come from signers seen in training. Accuracy for new signers is not established."],
    ["Calibration", "Scores overstate accuracy (mean top-1 score 35.6% vs 18.9% correct), so they are estimates, not probabilities."],
    ["Power", "1.42 W per-process CPU energy estimate, dominated by MediaPipe; not total device power and not measured on a phone."],
    ["Testing scope", "Webcam use verified only in two recordings (Windows, Brave); the live HUD is untested with a camera; nothing measured on a phone."],
    ["Scope and platform", "Isolated words only, not continuous translation. Python web app and macOS tools; no Android or iOS app."],
  ];
  limits.forEach(([head, body], i) => {
    const x = 0.6 + (i % 2) * 6.15;
    const y = 1.45 + Math.floor(i / 2) * 1.75;
    card(s, `limit ${i + 1}`, x, y, 5.95, 1.55);
    text(s, head, { objectName: `limit ${i + 1} head`, x: x + 0.25, y: y + 0.12, w: 5.4, h: 0.45, fontSize: 17, bold: true, color: C.accent3 });
    text(s, body, { objectName: `limit ${i + 1} body`, x: x + 0.25, y: y + 0.55, w: 5.5, h: 0.95, fontSize: 13, color: C.text1, valign: "top" });
  });
  s.addNotes("Be explicit about every failing or unverified item.");

  // 19. Future work
  s = content("Outlook", "Future improvements");
  const future = [
    ["More data per word", "Train on the full official training split (14,296 clips, 5-30 per word) with class-balanced sampling."],
    ["Signer-independent test", "Hold out entire signers for validation and test to measure generalisation honestly."],
    ["Lower power", "Run MediaPipe at a lower rate when no person is in view (measured 183 mW floor), or use the lighter pose model."],
    ["Phone deployment", "Package the ONNX model with ONNX Runtime Mobile and measure latency and power on a real device."],
    ["Public demo and live tests", "Deploy the prepared Hugging Face Space, verify webcam recording, and report results on new signers at angles and in dim light."],
  ];
  future.forEach(([head, body], i) => {
    const y = 1.45 + i * 1.02;
    iconDot(s, head, icons.road, 0.6, y + 0.08, 0.6);
    text(s, head, { objectName: `future ${i + 1} head`, x: 1.45, y: y + 0.05, w: 3.6, h: 0.65, fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    text(s, body, { objectName: `future ${i + 1} body`, x: 5.1, y: y + 0.05, w: 7.6, h: 0.75, fontSize: 14, color: C.text1, valign: "middle" });
  });
  s.addNotes("Proposals only; none of these have been run. Do not quote accuracy gains as results.");

  // 20. Conclusion
  s = pres.addSlide({ masterName: "Dark", sectionTitle: "Outlook" });
  n += 1;
  s.addText("Conclusion", { placeholder: "title" });
  text(s, "A complete, local, landmark-based isolated ASL word recognizer: MediaPipe extraction, shared preprocessing, a 0.87M-parameter 1D-CNN, ONNX FP32 and INT8 export, a browser demo app, an offline predict() API and a live caption HUD.", { objectName: "conclusion 1", x: 0.8, y: 3.85, w: 11.7, h: 1.1, fontSize: 20, color: "CFE3E5", valign: "top" });
  text(s, "Size, latency, drift, occlusion and privacy targets are met. Accuracy (18.90% top-1, 42.60% top-5) and power are not; the next steps are more training data per word and a signer-independent evaluation.", { objectName: "conclusion 2", x: 0.8, y: 5.05, w: 11.7, h: 1.1, fontSize: 20, color: "CFE3E5", valign: "top" });
  s.addNotes("Close on the honest summary.");

  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log(`wrote ${OUT} (${n} slides)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
