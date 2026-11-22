# Demo video: script, storyboard and recording guide

**Status:** a first version exists: `video/EchoGesture_Demo.mp4` (2:22). It combines two real webcam screen recordings of the web app with captioned slides (`video/make_demo_video.py`). It has no voice-over, no live HUD segment and no angled or partly hidden example. Use this guide to record a fuller version.

## Can the interactive demo replace the video?

**No evidence that it can.** The only organiser-side document found is the track solution guide (`EchoGesture_Solution_Guide.pdf`, page 25). It lists a separate **"Demo (2–3 min)"** with:
- live signing that produces captions on screen;
- an angled and a partly hidden example;
- an FPS and latency overlay;
- a mention of "no cloud" and the model size.

The original competition slides were not found. So **record the video**, and give the interactive demo link as an extra.

## Before recording

1. Create the target folder and launch the app:
   ```bash
   cd ~/hack/EchoGesture && source .venv/bin/activate
   mkdir -p video                      # the final file goes to video/EchoGesture_Demo.mp4
   python app.py                       # browser app on http://127.0.0.1:7860
   ```
2. Allow camera access for your browser, and for Terminal (needed for `hud_demo.py`), in **System Settings → Privacy & Security → Camera**.
3. Record the screen with **Cmd + Shift + 5 → Record Selected Portion** (macOS built-in) at 1920 × 1080 if possible. Export or rename the file to `video/EchoGesture_Demo.mp4`.
4. Prepare **your own** short sign clips (do not show WLASL videos in a public video; the dataset licence is academic use only). Good choices:
   - a sign recorded front-on;
   - one at about 30° to the camera;
   - one with a hand partly covered.
5. Practise once. Expect wrong predictions (test top-1 is 18.9%) and show them honestly, pointing to the top-5 list.

## Storyboard (target 2:30)

| Time | On screen | Narration (suggested) |
|---|---|---|
| 0:00–0:12 | Slide 1 (title), from `presentation/EchoGesture_Presentation.pdf` | "EchoGesture recognises isolated American Sign Language words, on the device, from body and hand landmarks." |
| 0:12–0:27 | Slide 2 (problem) | "One sign per clip, 2,000 possible words, no video leaving the device, and answers fast enough for conversation." |
| 0:27–0:45 | Slides 4 and 9 (pipeline, 1D-CNN) | "MediaPipe turns each frame into 75 landmarks. A 3.3 MB temporal 1D-CNN classifies their motion." |
| 0:45–1:15 | **App: upload.** Drag in your front-on clip, press *Recognise sign*, then point at the word, the top-5 bars and the timing table | "The model itself takes under a millisecond; most of the time is MediaPipe landmark extraction. Scores are estimates, not probabilities." |
| 1:15–1:40 | **App: webcam.** Click the camera icon, record one sign, submit, show the result | "The same pipeline runs on a webcam recording in the browser." |
| 1:40–2:00 | **Live HUD:** `python hud_demo.py`. Sign 2–3 words, including the angled and partly covered examples; keep the FPS and latency overlay visible | "In live mode signs are segmented automatically. Sign-end-to-word latency is about 32 ms at p95, plus a deliberate 320 ms wait to confirm the hands have dropped." |
| 2:00–2:20 | Slides 15 and 17 (benchmarks, scorecard) | "0.92 MB INT8 and 3.33 MB FP32, no cloud. Accuracy is 18.9% top-1 and power is about 1.4 W of process CPU, so both targets are missed." |
| 2:20–2:30 | Slide 20 (conclusion) | "Next: more training data per word and an evaluation on new signers." |

## What not to do

- Do not show fabricated predictions, edited results or WLASL clips.
- Do not describe the system as continuous translation, or claim the 70% accuracy target is met.
- Do not cut out wrong predictions to imply higher accuracy.

## After recording

1. Save the file as `video/EchoGesture_Demo.mp4` (2–3 minutes).
2. Update the "Video" row in `docs/Deliverables_Status.md` and the README link table.
3. If the file is larger than about 50 MB, upload it where the competition asks (or YouTube, unlisted) and link it instead of committing it.
