# Head Sway

## Metadata

- **Indicator key:** `head_sway_max_pct`
- **Label:** Head sway
- **Plain-language name:** How much the head moves sideways
- **Swing phase:** Backswing and overall swing
- **Unit:** Percent of body height

## Measurement definition

Measures the greatest sideways movement of the head during the swing, expressed relative to the golfer’s body height.

## Beginner explanation

Head sway describes how far the head moved side to side during the swing.

## Why it matters

This measurement helps describe head movement during the swing. It does not by itself determine balance, contact quality, swing consistency, or whether the golfer should keep the head still.

## Evidence classification

- **Directly measured:** Maximum sideways head movement during the swing, relative to body height.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Balance, stability, low-point control, contact quality, consistency, ball flight, posture, or whether the golfer should move the head more or less.

## Approved interpretation language

### Inside the reference range

The measured sideways head movement was within the tour reference range.

### Below the reference range

The measured sideways head movement was below the tour reference range.

### Above the reference range

The measured sideways head movement was above the tour reference range.

## Approved glossary

- **Head sway:** Sideways movement of the head during the swing.
- **Sideways movement:** Movement across the camera view rather than up or down.
- **Percent of body height:** A body-size-adjusted way of expressing movement distance.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured sideways head movement increased compared with the earlier swing.
- The measured sideways head movement decreased compared with the earlier swing.
- The measured sideways head movement was similar in both swings.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It measures sideways image-plane movement and may not represent full three-dimensional head motion.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera angle, camera movement, body visibility, pose-estimation quality, and head-joint detection.
- The value is normalized by body height to support comparison across golfers.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured amount of sideways head movement.
- State whether the measurement was inside, below, or above the tour reference range.
- Compare reliable head-sway measurements between two swings.
- Explain the measurement in beginner-friendly language.
- State that the value is relative to body height.
- Describe the movement neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Say the golfer kept the head “still enough,” moved it “too much,” or should keep it fixed.
- Describe the movement as correct, incorrect, good, bad, ideal, unstable, or faulty solely from the reference comparison.
- Claim that the measured head movement caused poor contact, inconsistent shots, a changing low point, a slice, a hook, or loss of distance.
- Infer balance, weight transfer, posture, eye position, spine movement, or body sway from this metric alone.
- Diagnose a physical limitation, vision issue, mobility issue, or injury.
- Recommend a drill, correction, head position, or movement target.
- Treat the tour reference range as the correct target for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured sideways head movement was within the tour reference range. In plain language, the amount your head moved side to side fell within the comparison range.

### Below the reference range

Your measured sideways head movement was below the tour reference range. This means the head moved a smaller distance side to side than the comparison range in this measurement.

### Above the reference range

Your measured sideways head movement was above the tour reference range. This means the head moved farther side to side than the comparison range in this measurement.

### Comparison — increased

Your measured sideways head movement increased compared with the earlier swing. This describes a difference between the two swings without by itself showing whether the current movement was better or worse.

### Comparison — decreased

Your measured sideways head movement decreased compared with the earlier swing. This shows how the measured head motion changed without judging that change as beneficial.

### Low confidence

The available camera view was not reliable enough to assess sideways head movement confidently for this swing.

## Prohibited response examples

- “You moved your head too much.”
- “Keep your head still.”
- “Your head sway caused the poor contact.”
- “Your head movement shows that your balance is weak.”
- “This is the ideal amount of head movement.”
- “Your head stayed perfectly stable.”
- “Your sway caused the ball to slice.”