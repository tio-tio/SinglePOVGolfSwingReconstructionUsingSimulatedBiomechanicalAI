# Posture Change from Setup to Impact

## Metadata

- **Indicator key:** `posture_loss_deg`
- **Label:** Posture change
- **Plain-language name:** How much the upper-body angle changes from setup to impact
- **Swing phase:** Setup to impact
- **Unit:** Degrees

## Measurement definition

Measures how much the golfer’s spine angle changed between the setup position and impact.

## Beginner explanation

Posture change describes how much the upper body’s forward-leaning angle changed during the swing.

## Why it matters

This measurement helps describe whether the golfer’s upper-body angle stayed similar or changed between setup and impact. It does not by itself identify a swing fault, explain why the change occurred, or determine contact quality.

## Evidence classification

- **Directly measured:** The change in spine angle from setup to impact, in degrees.
- **Derived comparison:** Whether the amount of change was inside or above the tour reference range.
- **Not supported:** Early extension, swing plane, balance, contact quality, flexibility, injury, ball flight, or whether the golfer should maintain more posture.

## Approved interpretation language

### Inside the reference range

The measured change in upper-body angle from setup to impact was within the tour reference range.

### Below the reference range

A below-range interpretation is not used for this metric unless the scorecard explicitly supports it.

### Above the reference range

The measured upper-body angle changed more from setup to impact than the tour reference range.

## Approved glossary

- **Posture:** The upper-body and spine angle held during the swing.
- **Posture change:** The difference in upper-body angle between setup and impact.
- **Spine angle:** How much the upper body leans away from an upright position.
- **Setup:** The golfer’s starting position before the swing begins.
- **Impact:** The moment the club meets the ball.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured posture change increased compared with the earlier swing.
- The measured posture change decreased compared with the earlier swing.
- The measured posture change was similar in both swings.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- It depends on reliable detection of both setup and impact.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera angle, body visibility, pose-estimation quality, and event detection.
- The metric reports the amount of angle change, not the direction or cause of the change unless that information is separately measured.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured amount of upper-body angle change from setup to impact.
- State whether the measurement was inside or above the tour reference range.
- Compare reliable posture-change measurements between two swings.
- Explain the measurement in beginner-friendly language.
- Describe the change neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Automatically call the movement “early extension.”
- Say the golfer stood up, dipped, lost posture, or stayed in posture unless a separate directional measurement supports that wording.
- Describe the movement as correct, incorrect, good, bad, ideal, or faulty solely from the reference comparison.
- Claim that the change caused poor contact, thin shots, fat shots, inconsistency, a slice, a hook, or loss of distance.
- Infer swing plane, pelvis movement, knee action, head movement, balance, or weight transfer from this metric alone.
- Diagnose mobility, flexibility, strength, posture problems, or injury.
- Recommend a drill, correction, spine angle, or posture target.
- Treat the tour reference range as the correct target for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured upper-body angle change from setup to impact was within the tour reference range. In plain language, the amount your posture changed fell within the comparison range.

### Above the reference range

Your measured upper-body angle changed more from setup to impact than the tour reference range. This describes the difference between the two measured positions without identifying a fault or cause.

### Comparison — increased

Your measured posture change increased compared with the earlier swing. That means the difference between your setup and impact angles was larger in the current swing.

### Comparison — decreased

Your measured posture change decreased compared with the earlier swing. That means the upper-body angle stayed more similar between setup and impact than in the earlier swing, without by itself showing that the change was better.

### Low confidence

The available camera view was not reliable enough to assess posture change confidently for this swing.

## Prohibited response examples

- “You lost your posture.”
- “You stood up through impact.”
- “This is early extension.”
- “Your posture change caused poor contact.”
- “You need to maintain your spine angle.”
- “Your posture was perfect.”
- “This movement shows limited mobility.”