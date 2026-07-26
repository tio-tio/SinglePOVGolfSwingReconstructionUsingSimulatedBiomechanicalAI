# Trail-Arm Position at the Top

## Metadata

- **Indicator key:** `right_arm_bend_top_deg`
- **Label:** Trail-arm position
- **Plain-language name:** How much the trail arm bends at the top
- **Swing phase:** Top of the backswing
- **Unit:** Degrees
- **Handedness note:** The current indicator key measures the right arm. The right arm is the trail arm for a right-handed golfer, but the lead arm for a left-handed golfer.

## Measurement definition

Measures the angle of the right arm at the top of the backswing to describe how much it appears folded or extended.

## Beginner explanation

Trail-arm position describes how much the arm farther from the target bends at the top of the backswing.

For a right-handed golfer, the trail arm is usually the right arm. For a left-handed golfer, the trail arm is usually the left arm.

## Why it matters

This measurement helps describe the shape and position of the measured arm at the top of the backswing. It does not by itself determine backswing quality, club support, power, swing plane, or whether the arm should bend more or less.

## Evidence classification

- **Directly measured:** The right-arm bend angle at the top of the backswing.
- **Derived comparison:** Whether the value was inside, below, or above the tour reference range.
- **Not supported:** Power, club support, swing width, sequencing, flexibility, injury, contact quality, ball flight, or whether the golfer should change the arm position.

## Approved interpretation language

### Inside the reference range

The measured right-arm bend at the top was within the tour reference range.

### Below the reference range

The measured right-arm bend was below the tour reference range, meaning the arm appeared less folded than the comparison range.

### Above the reference range

The measured right-arm bend was above the tour reference range, meaning the arm appeared more folded than the comparison range.

## Approved glossary

- **Trail arm:** The arm farther from the target.
- **Arm bend:** How much the elbow joint is folded.
- **Folded arm:** An arm position with more bend at the elbow.
- **Top of the backswing:** The point where the backswing ends and the downswing begins.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured right-arm bend increased compared with the earlier swing.
- The measured right-arm bend decreased compared with the earlier swing.
- The measured right-arm position was similar in both swings.
- The right arm appeared more folded than in the earlier swing.
- The right arm appeared less folded than in the earlier swing.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- Arm-angle estimates can be affected by camera angle, elbow visibility, clothing, body overlap, club position, and pose-estimation quality.
- This metric may be unreliable when the arm is partly hidden behind the torso or another body part.
- It should not be interpreted, compared, or revealed when the scorecard marks the measurement as low confidence.
- The current indicator key measures the right arm. Handedness must be known before calling it the trail arm.
- Use “right arm” when handedness is unknown.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured amount of right-arm bend at the top.
- State whether the reliable measurement was inside, below, or above the tour reference range.
- Say that a higher bend value represents a more folded measured arm when the metric definition supports that interpretation.
- Say that a lower bend value represents a less folded measured arm.
- Compare reliable right-arm measurements between two swings.
- Explain trail arm and arm bend in beginner-friendly language.
- State that the metric cannot be assessed when confidence is low.
- Use “right arm” instead of “trail arm” when handedness is unknown.

## Prohibited inferences

The LLM must not:

- Assume the golfer is right-handed unless handedness is available.
- Call the right arm the trail arm for a left-handed golfer.
- Say the arm was too folded, not folded enough, too straight, or ideally positioned.
- Describe the movement as correct, incorrect, good, bad, ideal, strong, weak, or faulty solely from the reference comparison.
- Claim that the arm position supported or failed to support the club.
- Claim that the arm position created or reduced power, speed, width, consistency, or contact quality.
- Infer wrist position, club position, shoulder turn, swing plane, elbow direction, or lead-arm movement from this metric alone.
- Diagnose flexibility, strength, elbow problems, mobility, or injury.
- Recommend changing the arm position, elbow bend, or backswing.
- Treat the tour reference range as the correct target for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured right-arm bend at the top was within the tour reference range. In plain language, the amount the arm folded fell within the comparison range.

### Below the reference range

Your measured right-arm bend was below the tour reference range at the top. This means the arm appeared less folded than the comparison range in this measurement.

### Above the reference range

Your measured right-arm bend was above the tour reference range at the top. This means the arm appeared more folded than the comparison range in this measurement.

### Comparison — more folded

Your measured right arm appeared more folded than in the earlier swing. This describes a change in arm position without showing by itself that the current swing was better.

### Comparison — less folded

Your measured right arm appeared less folded than in the earlier swing. This shows how the two measurements differed without judging the change as beneficial or harmful.

### Handedness unknown

The system measured the right arm at the top of the backswing. I cannot confidently call it the trail arm unless the golfer’s handedness is known.

### Low confidence

The available camera view was not reliable enough to assess the right-arm position confidently for this swing.

## Prohibited response examples

- “Your trail arm folded too much.”
- “You need to tuck your right elbow.”
- “Your right arm did not support the club.”
- “This arm position cost you power.”
- “Your elbow position caused the slice.”
- “Your trail arm was perfect.”
- “Your elbow mobility is limited.”
- “You need more bend in the trail arm.”