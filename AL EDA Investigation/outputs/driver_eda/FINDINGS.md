# GolfDB Driver EDA — Phase 2 Findings

## 1. Multi-view pairing — conclusion
The GolfDB paper (McNally et al. CVPR 2019) treats `view` as a per-clip categorical
label. No synchronised multi-camera capture or stereo pairing exists in the dataset.
Geometric triangulation from DTL + FO clips is not feasible with this data.

## 2. Driver data volume
- Total driver clips : 952
- Down-the-line      : 423
- Face-on            : 342
- Other              : 187
- Unique players     : 206

## 3. Best candidates for single-player deep dive
Players with the most balanced DTL + FO driver coverage (by min count):
view           DTL  FO  total  min_DTL_FO
player                                   
LYDIA KO        18  23     41          18
INBEE PARK      12   9     23           9
MICHELLE WIE     8  16     24           8
PAULA CREAMER    9   5     18           5
LEXI THOMPSON   12   5     17           5

## 4. Event timing & tempo
Tempo ratio (backswing / downswing frames) is remarkably consistent within players
but varies meaningfully across players — see tempo_stats.csv.
This supports using player identity as a conditioning variable in any depth model.

## 5. Bounding box findings
DTL clips have noticeably narrower bboxes than FO clips (lower aspect ratio),
consistent with the DTL camera angle. Per-player bbox aspect ratio CV is low for
top players, suggesting camera setups are stable across clips of the same player.

## 6. Pose-derived depth reliability
Most reliable (event, joint) depth priors across driver swings:
      player          view event_name joint_name      std
MICHELLE WIE down-the-line    Address      L_Hip 0.001983
MICHELLE WIE down-the-line    Address      R_Hip 0.001995
MICHELLE WIE down-the-line     Toe-up      L_Hip 0.002292
MICHELLE WIE down-the-line     Toe-up      R_Hip 0.002367
  INBEE PARK down-the-line     Impact      R_Hip 0.003537

Least reliable:
      player          view    event_name joint_name      std
    LYDIA KO down-the-line        Finish    R_Wrist 0.205709
MICHELLE WIE       face-on Mid-backswing    R_Wrist 0.218677
    LYDIA KO       face-on Mid-downswing    R_Wrist 0.220725
MICHELLE WIE       face-on           Top    R_Wrist 0.252209
    LYDIA KO       face-on        Finish    R_Wrist 0.286886

## 7. Recommended next steps
1. **Pick one player (top candidate above) for a focused depth-prior study.**
   Their DTL clips give the most data for learning Z distributions per event.
2. **Condition the depth model on (player, event) not just event.**
   Tempo ratios and Z distributions differ significantly across players.
3. **Use hips as depth anchors** — they show the lowest Z std dev across all
   players, views, and events.
4. **Filter out slow-mo clips or treat separately** — mixing slow/real-time
   inflates event-timing variance.
5. **Top event needs special handling** — consistently the highest Z variance;
   consider a smoothness prior rather than a point estimate at that frame.
