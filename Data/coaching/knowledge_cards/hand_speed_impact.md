# Hand Speed into Impact

## Metadata

- **Indicator key:** `hand_speed_impact_bs`
- **Label:** Hand speed into impact
- **Plain-language name:** How quickly the hands move through the impact area
- **Swing phase:** Impact
- **Unit:** Body-lengths per second

## Measurement definition

Measures the peak speed of the hands through the impact area, normalized relative to the golfer’s body size.

## Beginner explanation

Hand speed describes how quickly the golfer’s hands moved as they approached and passed through the impact area.

The value is adjusted for body size so that measurements can be compared more consistently across golfers.

## Why it matters

This measurement helps describe the speed of the hands through one part of the swing. It does not directly measure clubhead speed, ball speed, power, carry distance, contact quality, or shot outcome.

## Evidence classification

- **Directly measured:** Peak hand speed through the impact area, normalized relative to body size.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Clubhead speed, ball speed, power, efficiency, carry distance, contact quality, ball flight, or whether the golfer should move the hands faster or slower.

## Approved interpretation language

### Inside the reference range

The measured hand speed through the impact area was within the tour reference range.

### Below the reference range

The measured hand speed through the impact area was below the tour reference range.

### Above the reference range

The measured hand speed through the impact area was above the tour reference range.

## Approved glossary

- **Hand speed:** How quickly the hands travel during the swing.
- **Impact area:** The part of the swing immediately around the moment the club reaches the ball.
- **Peak speed:** The highest measured speed during the selected part of the swing.
- **Body-lengths per second:** A body-size-adjusted speed unit that expresses movement relative to the golfer’s height or scale.
- **Normalized by body size:** Adjusted relative to the golfer’s body dimensions to support comparison across golfers.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured hand speed increased compared with the earlier swing.
- The measured hand speed decreased compared with the earlier swing.
- The measured hand speed was similar in both swings.
- The hands moved faster through the impact area than in the earlier swing.
- The hands moved slower through the impact area than in the earlier swing.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It depends on reliable hand-joint tracking, frame timing, body-scale estimation, and impact detection.
- It should not be interpreted, compared, or revealed when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera frame rate, camera angle, motion blur, hand visibility, body overlap, pose-estimation quality, and missing frames.
- The measurement describes hand speed rather than clubhead speed or ball speed.
- The normalized unit may not correspond directly to miles per hour or kilometres per hour.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured speed of the hands through the impact area.
- State whether the reliable measurement was inside, below, or above the tour reference range.
- Compare reliable hand-speed measurements between two swings.
- Say that the hands moved faster or slower than in an earlier measured swing.
- Explain the body-size-adjusted unit in beginner-friendly language.
- Distinguish hand speed from clubhead speed and ball speed.
- Describe the measurement neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Call the measurement clubhead speed, swing speed, or ball speed.
- Convert the value into miles per hour or kilometres per hour unless an approved conversion method exists.
- Say the golfer’s hands were too fast, too slow, fast enough, or not fast enough.
- Describe the measurement as correct, incorrect, good, bad, ideal, powerful, weak, efficient, or faulty solely from the reference comparison.
- Claim that higher hand speed produced more power, clubhead speed, ball speed, carry distance, or better contact.
- Claim that lower hand speed caused a loss of distance or poor contact.
- Predict carry, trajectory, launch, slice, hook, fade, draw, or another shot result.
- Infer wrist action, release timing, club position, sequencing, acceleration pattern, or impact quality from this metric alone.
- Recommend speeding up, slowing down, releasing the club differently, or using a drill.
- Treat the tour reference range as the correct target for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured hand speed through the impact area was within the tour reference range. In plain language, the speed of your hands fell within the comparison range.

### Below the reference range

Your measured hand speed through the impact area was below the tour reference range. This describes how quickly the hands moved in this swing and does not directly measure clubhead speed or distance.

### Above the reference range

Your measured hand speed through the impact area was above the tour reference range. This means the hands moved faster than the comparison range in this measurement, without by itself showing the resulting clubhead or ball speed.

### Comparison — increased

Your measured hand speed increased compared with the earlier swing. The hands moved faster through the impact area in the current measurement, but this does not by itself establish more power or distance.

### Comparison — decreased

Your measured hand speed decreased compared with the earlier swing. This describes a change in hand movement without by itself showing whether the current swing was better or worse.

### Explaining the unit

This hand-speed value is expressed relative to body size rather than in miles per hour. It allows the movement to be compared across golfers of different sizes.

### Low confidence

The available video and hand tracking were not reliable enough to assess hand speed confidently for this swing.

## Prohibited response examples

- “Your clubhead speed was below average.”
- “Your hands were too slow.”
- “This hand speed cost you distance.”
- “Your faster hands created more power.”
- “You need to release the club faster.”
- “Your hand speed caused the slice.”
- “This is the ideal impact speed.”
- “Your swing speed was excellent.”