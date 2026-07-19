# Head Height Change

## Metadata

- **Indicator key:** `head_lift_max_pct`
- **Label:** Head height change
- **Plain-language name:** How much the head moves up and down
- **Swing phase:** Downswing and overall swing
- **Unit:** Percent of body height

## Measurement definition

Measures the greatest upward or downward movement of the head during the swing, expressed relative to the golfer’s body height.

## Beginner explanation

Head height change describes how much the head moved vertically during the swing.

## Why it matters

This measurement helps describe vertical head movement. It does not by itself determine posture quality, contact consistency, balance, swing plane, or whether the golfer should keep the head at a fixed height.

## Evidence classification

- **Directly measured:** Maximum vertical head movement during the swing, relative to body height.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Contact quality, low-point control, posture quality, balance, swing plane, ball flight, or whether the golfer should move the head more or less.

## Approved interpretation language

### Inside the reference range

The measured head height change was within the tour reference range.

### Below the reference range

The measured head height change was below the tour reference range.

### Above the reference range

The measured head height change was above the tour reference range.

## Approved glossary

- **Head height change:** Upward or downward movement of the head during the swing.
- **Vertical movement:** Movement up or down in the camera view.
- **Percent of body height:** A body-size-adjusted way of expressing movement distance.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured head height change increased compared with the earlier swing.
- The measured head height change decreased compared with the earlier swing.
- The measured head height change was similar in both swings.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It measures vertical movement in the camera image and may not represent full three-dimensional head motion.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera angle, camera movement, body visibility, pose-estimation quality, and head-joint detection.
- The value is normalized by body height to support comparison across golfers.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured amount of vertical head movement.
- State whether the measurement was inside, below, or above the tour reference range.
- Compare reliable head-height measurements between two swings.
- Explain the measurement in beginner-friendly language.
- State that the value is relative to body height.
- Describe the movement neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Say the golfer lifted, dipped, stood up, or stayed down unless the measurement direction specifically supports that wording.
- Say the golfer moved “too much,” “too little,” or should keep the head at one height.
- Describe the movement as correct, incorrect, good, bad, ideal, unstable, or faulty solely from the reference comparison.
- Claim that the movement caused thin contact, fat contact, topped shots, poor consistency, a changing low point, or another shot result.
- Infer spine-angle change, posture loss, knee movement, balance, or body sway from this metric alone.
- Diagnose mobility, strength, vision, posture, or injury.
- Recommend a drill, correction, head position, or movement target.
- Treat the tour reference range as the correct target for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured head height change was within the tour reference range. In plain language, the amount your head moved up and down fell within the comparison range.

### Below the reference range

Your measured head height change was below the tour reference range. This means the head moved a smaller vertical distance than the comparison range in this measurement.

### Above the reference range

Your measured head height change was above the tour reference range. This means the head moved farther up or down than the comparison range in this measurement.

### Comparison — increased

Your measured head height change increased compared with the earlier swing. This describes a difference in vertical head movement without by itself showing whether the current swing was better or worse.

### Comparison — decreased

Your measured head height change decreased compared with the earlier swing. This shows how the measured head movement changed without judging that change as beneficial.

### Low confidence

The available camera view was not reliable enough to assess head height change confidently for this swing.

## Prohibited response examples

- “You lifted your head too early.”
- “Keep your head down.”
- “Your head movement caused the topped shot.”
- “You stood up through impact.”
- “Your vertical movement shows poor posture.”
- “This is the ideal amount of head movement.”
- “Your head stayed perfectly level.”