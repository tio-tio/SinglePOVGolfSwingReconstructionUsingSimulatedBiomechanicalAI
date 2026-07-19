# Spine Angle at Impact

## Metadata

- **Indicator key:** `spine_tilt_impact_deg`
- **Label:** Spine angle at impact
- **Plain-language name:** How much the upper body leans at impact
- **Swing phase:** Impact
- **Unit:** Degrees

## Measurement definition

Measures how far the upper body is tilted away from an upright position at the moment of impact.

## Beginner explanation

Spine angle at impact describes how much the golfer’s upper body was leaning when the club reached the ball.

## Why it matters

This measurement helps describe the golfer’s upper-body position at impact. It does not by itself determine posture quality, swing plane, contact quality, or why the golfer reached that position.

## Evidence classification

- **Directly measured:** Upper-body tilt at impact, in degrees.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Posture quality, swing plane, early extension, balance, contact quality, ball flight, physical limitations, or whether the golfer should lean more or less.

## Approved interpretation language

### Inside the reference range

The measured upper-body angle at impact was within the tour reference range.

### Below the reference range

The measured upper-body angle at impact was below the tour reference range, meaning the golfer was more upright at impact than the comparison range.

### Above the reference range

The measured upper-body angle at impact was above the tour reference range, meaning the golfer showed more upper-body lean at impact than the comparison range.

## Approved glossary

- **Spine angle:** How much the upper body leans away from an upright position.
- **Impact:** The moment the club meets the ball.
- **Upper-body lean:** The angle of the torso relative to standing upright.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured spine angle at impact increased compared with the earlier swing.
- The measured spine angle at impact decreased compared with the earlier swing.
- The measured impact angle was similar in both swings.
- The current impact position was more upright than the earlier impact position.
- The current impact position showed more upper-body lean than the earlier impact position.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It depends on reliable detection of impact and the relevant body joints.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera angle, camera tilt, body visibility, pose-estimation quality, clothing, stance orientation, and impact detection.
- The metric describes upper-body angle in the available view and may not capture the golfer’s full three-dimensional posture.
- This metric measures the angle at impact, not how much the angle changed from address. Use `posture_loss_deg` for the setup-to-impact change.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured upper-body angle at impact.
- State whether the measurement was inside, below, or above the tour reference range.
- Say that a lower value represents a more upright measured impact position.
- Say that a higher value represents more measured upper-body lean.
- Compare reliable impact-angle measurements between two swings.
- Explain spine angle and impact in beginner-friendly language.
- Describe the measured position neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Say the golfer stood up, stayed down, maintained posture, or lost posture based only on the impact angle.
- Call the movement early extension unless a separate approved metric directly supports that conclusion.
- Describe the impact position as correct, incorrect, good, bad, ideal, athletic, stable, or faulty solely from the reference comparison.
- Claim that the angle caused thin contact, fat contact, topped shots, a slice, a hook, loss of distance, or another shot result.
- Infer the direction or amount of change from address without retrieving `spine_tilt_address_deg` or `posture_loss_deg`.
- Infer knee flex, hip movement, head movement, balance, weight transfer, club path, clubface angle, or equipment fit from this metric alone.
- Diagnose mobility, flexibility, strength, back problems, posture problems, or injury.
- Recommend a target angle, posture change, drill, or correction.
- Treat the tour reference range as the correct impact position for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured upper-body angle at impact was within the tour reference range. In plain language, the amount of upper-body lean at impact fell within the comparison range.

### Below the reference range

Your measured upper-body angle at impact was below the tour reference range. This means your upper body appeared more upright at impact than the comparison range in this measurement.

### Above the reference range

Your measured upper-body angle at impact was above the tour reference range. This means your upper body showed more lean at impact than the comparison range in this measurement.

### Comparison — increased

Your measured spine angle at impact increased compared with the earlier swing. That means the current impact position showed more upper-body lean.

### Comparison — decreased

Your measured spine angle at impact decreased compared with the earlier swing. That means the current impact position appeared more upright.

### Distinguishing impact angle from posture change

This metric describes your upper-body angle at impact. It does not by itself show how much your posture changed from setup.

### Low confidence

The available camera view was not reliable enough to assess your upper-body angle at impact confidently for this swing.

## Prohibited response examples

- “You stood up through impact.”
- “You maintained your posture perfectly.”
- “This is early extension.”
- “Your impact posture caused the poor contact.”
- “You need more spine tilt at impact.”
- “Your impact position is ideal.”
- “Your back mobility is limiting your swing.”
- “This angle caused the ball to slice.”