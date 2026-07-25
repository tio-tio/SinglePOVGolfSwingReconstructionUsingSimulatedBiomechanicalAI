# Hip Turn at the Top

## Metadata

- **Indicator key:** `hip_turn_top_deg`
- **Label:** Hip turn at the top
- **Plain-language name:** How far the hips rotate at the top
- **Swing phase:** Top of the backswing
- **Unit:** Degrees

## Measurement definition

Measures how far the hips rotated away from the target by the top of the backswing.

## Beginner explanation

Hip turn describes how much the lower body rotated during the backswing.

## Why it matters

This measurement helps describe how the lower body moved during the backswing. It can also be considered alongside shoulder turn when both measurements are available.

## Evidence classification

- **Directly measured:** Hip rotation at the top of the backswing, in degrees.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Power production, flexibility, mobility, balance, ball flight, injury, or whether the golfer should rotate more or less.

## Approved interpretation language

### Inside the reference range

The measured hip turn was within the tour reference range at the top of the backswing.

### Below the reference range

The measured hip turn was below the tour reference range at the top of the backswing.

### Above the reference range

The measured hip turn was above the tour reference range at the top of the backswing.

## Approved glossary

- **Hip turn:** How much the lower body rotates during the backswing.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.
- **Top of the backswing:** The point where the backswing ends and the downswing begins.

## Comparison language

- The measured hip turn increased compared with the earlier swing.
- The measured hip turn decreased compared with the earlier swing.
- The measured hip turn was similar in both swings.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It describes measured hip rotation but does not determine why the golfer moved this way.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera position, body visibility, pose-estimation quality, and the detected swing phase.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured amount of hip rotation.
- State whether the measurement was inside, below, or above the tour reference range.
- Compare reliable hip-turn measurements from two swings.
- Explain hip turn in beginner-friendly language.
- Describe the movement neutrally.
- Discuss shoulder turn separately when that measurement has also been retrieved.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Say the golfer turned “enough,” “not enough,” “too much,” or “too little.”
- Describe the movement as correct, incorrect, good, bad, ideal, or faulty solely from the reference comparison.
- Claim the hip turn created or reduced power.
- Diagnose flexibility, mobility, strength, balance, posture, or injury.
- Predict ball direction, contact quality, or shot outcome.
- Recommend a drill, correction, target angle, or amount of rotation.
- Treat the tour reference range as the correct target for every golfer.
- Infer shoulder–hip separation unless `x_factor_top_deg` was also retrieved.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured hip turn was within the tour reference range at the top of the backswing. In plain language, your lower body rotated by an amount that fell within the comparison range.

### Below the reference range

Your measured hip turn was below the tour reference range at the top of the backswing. This describes the movement captured in this swing rather than identifying a fault.

### Above the reference range

Your measured hip turn was above the tour reference range at the top of the backswing. That means the lower body rotated farther than the comparison range in this measurement.

### Comparison — increased

Your measured hip turn increased compared with the earlier swing. This describes how the two measured swings differed; it does not by itself mean the current movement was better.

### Comparison — decreased

Your measured hip turn decreased compared with the earlier swing. This shows a change between the two measurements without identifying whether that change was beneficial.

### Low confidence

The available camera view was not reliable enough to assess hip turn confidently for this swing.

## Prohibited response examples

- “You need to turn your hips more.”
- “Your hips did not rotate enough.”
- “You rotated your hips too far.”
- “This cost you power.”
- “Your hip turn caused the ball to slice.”
- “Your flexibility is limiting your turn.”
- “This is the ideal amount of hip turn.”