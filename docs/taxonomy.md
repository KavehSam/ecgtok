# Beat tokens

A cardiologist does not read a beat as a list of samples. The report names a few measurements: whether the QRS is wide, whether the R wave is small, whether the ST segment is elevated, whether the beat came early. The tokens are those names, one symbol each.

The window is MLII, 300 ms before the annotated R peak and 600 ms after, at 360 Hz, after a 0.6–40 Hz FIR bandpass. The R peak is the center of that window. QRS onset and offset are the stretch around the R peak where the derivative stays in the top 30 percent. Duration is that stretch in milliseconds.

| Measurement | Cut points |
|---|---|
| QRS duration | 80, 100, 120, 160 ms |
| R-wave amplitude, absolute | 0.15, 0.50, 0.80, 1.50 mV |
| ST level, 80 ms after QRS offset | −0.10 and +0.10 mV |
| T-wave amplitude | −0.10 and +0.10 mV; absent if no T peak is found |
| PR interval | 100, 120, 200, 240 ms; unmeasurable if no P onset |
| Prematurity | previous R–R divided by the median of up to ten earlier intervals: 0.70, 0.85, 1.15, 1.40 |

Polarity compares the positive and negative peaks inside the QRS. A notch is two peaks on the dominant side at least 20 ms apart. Concordance says whether the ST segment, or the T wave, points the same way as the QRS. The pause token uses the sum of the previous and next R–R intervals against the same baseline.

Each clause is wrapped as one token, for example `<QRS_WIDE>` or `<ST_ISOELECTRIC>`. The bins are coarse on purpose. They are the distinctions a reading already makes, not a reconstruction of the waveform.

The opaque control, added in the following change, replaces this string with energy bins of the same window. The language model, the split, and the labels stay the same, so a difference in score is a difference in the input.
