# Spine Angle at Address

## Metadata

- **Indicator key:** `spine_tilt_address_deg`
- **Label:** Spine angle at address
- **Plain-language name:** How much the upper body leans at setup
- **Swing phase:** Address
- **Unit:** Degrees

## Measurement definition

Measures how far the upper body is tilted away from an upright position at address.

## Beginner explanation

Spine angle at address describes how much the golfer’s upper body leans at setup before the swing begins.

## Why it matters

This measurement helps describe the golfer’s starting upper-body position. It does not by itself determine setup quality, balance, comfort, swing plane, or whether the golfer should lean more or less.

## Evidence classification

- **Directly measured:** Upper-body tilt at address, in degrees.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Setup quality, balance, mobility, comfort, club fit, swing plane, contact quality, ball flight, or whether the golfer should change the angle.

## Approved interpretation language

### Inside the reference range

The measured upper-body angle at address was within the tour reference range.

### Below the reference range

The measured upper-body angle at address was below the tour reference range, meaning the golfer began in a more upright position than the comparison range.

### Above the reference range

The measured upper-body angle at address was above the tour reference range, meaning the golfer began with more upper-body lean than the comparison range.

## Approved glossary

- **Spine angle:** How much the upper body leans away from an upright position.
- **Address:** The setup position immediately before the swing begins.
- **Upper-body lean:** The angle of the torso relative to standing upright.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured spine angle at address increased compared with the earlier swing.
- The measured spine angle at address decreased compared with the earlier swing.
- The measured setup angle was similar in both swings.
- The current setup was more upright than the earlier setup.
- The current setup had more upper-body lean than the earlier setup.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It depends on reliable detection of the address position and the relevant body joints.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera angle, camera tilt, body visibility, pose-estimation quality, clothing, and stance orientation.
- The metric describes upper-body angle in the available view and may not capture the golfer’s full three-dimensional posture.
- The tour range is a comparison reference, not an ideal setup target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured upper-body angle at address.
- State whether the measurement was inside, below, or above the tour reference range.
- Say that a lower value represents a more upright measured setup.
- Say that a higher value represents more measured upper-body lean.
- Compare reliable address-angle measurements between two swings.
- Explain address and spine angle in beginner-friendly language.
- Describe the starting position neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Say the golfer was standing too tall, bending too much, or had the correct setup solely from the reference comparison.
- Describe the setup as correct, incorrect, good, bad, ideal, athletic, balanced, comfortable, or faulty solely from this metric.
- Claim that the angle caused a slice, hook, poor contact, inconsistency, loss of distance, or another shot result.
- Infer knee flex, hip hinge, weight distribution, arm position, distance from the ball, club length, or equipment fit from this metric alone.
- Diagnose mobility, flexibility, strength, back problems, posture problems, or injury.
- Recommend a target angle, setup change, drill, or correction.
- Treat the tour reference range as the correct setup for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured upper-body angle at address was within the tour reference range. In plain language, the amount of forward lean in your setup fell within the comparison range.

### Below the reference range

Your measured upper-body angle at address was below the tour reference range. This means your setup appeared more upright than the comparison range in this measurement.

### Above the reference range

Your measured upper-body angle at address was above the tour reference range. This means your setup showed more upper-body lean than the comparison range in this measurement.

### Comparison — increased

Your measured spine angle at address increased compared with the earlier swing. That means the current setup showed more upper-body lean.

### Comparison — decreased

Your measured spine angle at address decreased compared with the earlier swing. That means the current setup appeared more upright.

### Low confidence

The available camera view was not reliable enough to assess your upper-body angle at address confidently for this swing.

## Prohibited response examples

- “You are standing too tall.”
- “You are bent over too much.”
- “Your setup posture is perfect.”
- “You need more spine tilt.”
- “This setup angle caused the slice.”
- “Your back is too rounded.”
- “Your mobility is limiting your address position.”
- “This is the ideal setup angle.”