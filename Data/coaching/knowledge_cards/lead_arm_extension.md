# Lead-Arm Extension at the Top

## Metadata

- **Indicator key:** `left_arm_bend_top_deg`
- **Label:** Lead-arm extension
- **Plain-language name:** How straight or bent the lead arm is at the top
- **Swing phase:** Top of the backswing
- **Unit:** Degrees
- **Handedness note:** The current indicator key assumes the left arm is the lead arm, which applies to a right-handed golfer.

## Measurement definition

Measures the angle of the left arm at the top of the backswing to describe how straight or bent it appears.

## Beginner explanation

Lead-arm extension describes how much the arm closer to the target is straightened or bent at the top of the backswing.

For a right-handed golfer, the lead arm is usually the left arm. For a left-handed golfer, the lead arm is usually the right arm.

## Why it matters

This measurement helps describe the shape and position of the lead arm at the top of the backswing. It does not by itself determine backswing quality, swing width, power, contact quality, or whether the arm should be straighter.

## Evidence classification

- **Directly measured:** The lead-arm bend angle at the top of the backswing.
- **Derived comparison:** Whether the value was inside or outside the tour reference range.
- **Not supported:** Power, swing width, structure, flexibility, injury, contact quality, ball flight, or whether the golfer should straighten or bend the arm more.

## Approved interpretation language

### Inside the reference range

The measured lead-arm position was within the tour reference range at the top of the backswing.

### Below the reference range

The measured lead-arm angle was below the tour reference range, indicating more bend than the comparison range.

### Above the reference range

An above-range interpretation should only be used if the scorecard explicitly supports one for this metric.

## Approved glossary

- **Lead arm:** The arm closer to the target.
- **Lead-arm extension:** How straight the lead arm appears during the swing.
- **Arm bend angle:** The measured angle used to describe how bent or straight the arm is.
- **Top of the backswing:** The point where the backswing ends and the downswing begins.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.

## Comparison language

- The measured lead-arm bend increased compared with the earlier swing.
- The measured lead-arm bend decreased compared with the earlier swing.
- The measured lead-arm position was similar in both swings.
- The lead arm appeared straighter than in the earlier swing.
- The lead arm appeared more bent than in the earlier swing.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.

## Confidence and camera limitations

- This is a motion estimate derived from the available camera view.
- Arm-angle estimates can be sensitive to camera angle, elbow visibility, clothing, body overlap, and pose-estimation quality.
- This metric is especially vulnerable to occlusion when the arm crosses the torso or club.
- It should not be interpreted, compared, or revealed when the scorecard marks the measurement as low confidence.
- The current indicator key assumes the left arm is the lead arm. Handedness should be confirmed before describing it as the lead arm.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured amount of arm bend at the top.
- State whether the reliable measurement was inside or below the tour reference range.
- Say that a smaller extension value represents more measured bend when the metric definition supports that interpretation.
- Compare reliable measurements between two swings.
- Explain lead arm and arm extension in beginner-friendly language.
- State that the metric cannot be assessed when confidence is low.
- Use “left arm” instead of “lead arm” when handedness is unknown.

## Prohibited inferences

The LLM must not:

- Assume the golfer is right-handed unless handedness is available.
- Describe the left arm as the lead arm for a left-handed golfer.
- Say the arm was too bent, not straight enough, or ideally straight.
- Describe the movement as correct, incorrect, good, bad, strong, weak, ideal, or faulty solely from the reference comparison.
- Claim that the arm position created or reduced power, width, speed, distance, consistency, or contact quality.
- Infer shoulder turn, wrist position, club position, swing plane, or trail-arm movement from this metric alone.
- Diagnose flexibility, strength, elbow problems, mobility, or injury.
- Recommend straightening the arm, changing the backswing, or using a drill.
- Treat the tour reference range as the correct target for every golfer.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured left-arm position at the top was within the tour reference range. In plain language, the amount of bend in the arm fell within the comparison range.

### Below the reference range

Your measured left-arm angle was below the tour reference range at the top. This means the arm appeared more bent than the comparison range in this measurement.

### Comparison — straighter

Your measured left arm appeared straighter than in the earlier swing. This describes a change in arm position without showing by itself that the current swing was better.

### Comparison — more bent

Your measured left arm appeared more bent than in the earlier swing. This shows how the two measurements differed without judging the change as beneficial or harmful.

### Handedness unknown

The system measured the left arm at the top of the backswing. I cannot confidently call it the lead arm unless the golfer’s handedness is known.

### Low confidence

The available camera view was not reliable enough to assess the arm position confidently for this swing.

## Prohibited response examples

- “Your lead arm was too bent.”
- “You need to keep your left arm straight.”
- “Your arm collapsed at the top.”
- “This cost you power.”
- “Your swing lacks width.”
- “Your elbow mobility is limited.”
- “This is the perfect lead-arm position.”
- “Your bent arm caused poor contact.”