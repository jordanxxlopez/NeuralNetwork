# Lecture 57 videos — Music and sound in animals

Downloadable MP4 files (H.264 video, AAC audio) that embed directly in PowerPoint,
Keynote and Google Slides, plus a script that inserts them into
`Neuroethology_Lecture57_FA2026.pptx` as video slides styled like the deck's own slides.

## Videos

| File | Shows | Source | Placed after slide |
|---|---|---|---|
| `00_Ronan_sea_lion_keeps_a_beat_UCSC.mp4` | Ronan the sea lion keeping the beat (supplied by instructor) | UC Santa Cruz; Cook et al. (2013) *J Comp Psychol* 127:412 | 21 → **21b** |
| `01_Snowball_cockatoo_Patel2009_MovieS1.mp4` | Snowball dancing to one song at several tempi | Patel et al. (2009) *Curr Biol* 19:827, Movie S1 | 16 → **16a** |
| `02_Budgerigar_tapping_Hasegawa2011_VideoS1.mp4` | Budgerigar key pecking to the metronome | Hasegawa et al. (2011) *Sci Rep* 1:120, Video S1 | 12 → **12a** |
| `03_Budgerigar_tapping_Hasegawa2011_VideoS2.mp4` | Second bird, same task | Hasegawa et al. (2011), Video S2 | 12 → **12b** |
| `04_Budgerigar_continuation_Hasegawa2011_VideoS3.mp4` | Pecking continues at a previous tempo | Hasegawa et al. (2011), Video S3 | 15 → **15a** |
| `05_Budgerigar_continuation_Hasegawa2011_VideoS4.mp4` | Second example of retained tempo | Hasegawa et al. (2011), Video S4 | 15 → **15b** |
| `06_Rats_music_Ito2022_MovieS1.mp4` | Rat movements during music (with slow motion) | Ito et al. (2022) *Sci Adv* 8:eabo7019, Movie S1 | 30 → **30a** |
| `07_Rat_bipedal_beat_Ito2022_MovieS2.mp4` | Beat-synchronous rat movement, bipedal stance | Ito et al. (2022), Movie S2 | 32 → **32a** |
| `08_Human_head_movement_Ito2022_MovieS3.mp4` | Human head movement to Mozart K.448 | Ito et al. (2022), Movie S3 | 33 → **33a** |
| `09_Chimpanzee_tapping_task_Hattori2013_VideoS1.mp4` | Chimpanzee keyboard tapping task | Hattori et al. (2013) *Sci Rep* 3:1566, Video S1 | 11 → **11a** |
| `10_Chimpanzee_Ai_tapping_Hattori2013_VideoS2.mp4` | Ai tapping with a 600 ms sound | Hattori et al. (2013), Video S2 | 11 → **11b** |
| `11_Chimpanzee_swaying_Hattori2020_MovieS1.mp4` | Chimpanzee swaying to sound | Hattori & Tomonaga (2020) *PNAS* 117:936, Movie S1 | 21 → **21a** |
| `12_Palm_cockatoo_drumming_Heinsohn2017_MovieS1.mp4` | Wild palm cockatoo drumming with a seedpod | Heinsohn et al. (2017) *Sci Adv* 3:e1602399, Movie S1 | 20 → **20a** |

That is 12 videos plus the Ronan video. The article videos are the papers' own
published supplementary files, downloaded from the PubMed Central open-access
collection (Patel et al. 2009 from the Current Biology open archive). They were
re-encoded only to H.264/AAC MP4; the three chimpanzee clips also had black
borders cropped.

## Insert them into the deck

```bash
python lecture_videos/insert_video_slides.py \
    Neuroethology_Lecture57_FA2026.pptx \
    lecture_videos/lecture57_video_slides.json \
    output/Neuroethology_Lecture57_FA2026_VIDEOS.pptx

python lecture_videos/verify_integrity.py \
    Neuroethology_Lecture57_FA2026.pptx \
    output/Neuroethology_Lecture57_FA2026_VIDEOS.pptx
```

`lecture57_video_slides.json` holds each slide's title, body text, caption,
footer citation, footer number (e.g. `16a`) and speaker notes.

How the new slides are built:
- Each new slide copies a figure slide from the same deck. Title, body text
  boxes, caption, footer citation and footer number keep the deck's own
  positions, fonts, sizes and colors. The figure is replaced by the embedded
  video, with a poster frame.
- The 46 original slides are not touched. The output copies every original part
  byte-for-byte and only adds new parts plus the new slide order.
  `verify_integrity.py` checks this.

Requires `python-pptx`, `lxml` and `ffmpeg`.
