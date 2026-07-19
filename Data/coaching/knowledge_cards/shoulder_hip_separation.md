# Shoulder–Hip Separation at the Top

## Metadata

- **Indicator key:** `x_factor_top_deg`
- **Label:** Shoulder–hip separation
- **Plain-language name:** The difference between shoulder turn and hip turn at the top
- **Swing phase:** Top of the backswing
- **Unit:** Degrees

## Measurement definition

Measures the difference between shoulder rotation and hip rotation at the top of the backswing.

## Beginner explanation

Shoulder–hip separation describes how much farther the shoulders rotated than the hips at the top of the backswing.

## Why it matters

This measurement helps describe the relationship between upper-body rotation and lower-body rotation at the top of the swing. It does not by itself determine power, efficiency, or swing quality.

## Evidence classification

- **Directly measured:** The difference between shoulder turn and hip turn at the top, in degrees.
- **Derived comparison:** Whether the separation value was inside, below, or above the tour reference range.
- **Not supported:** Power production, flexibility, mobility, injury risk, ball flight, swing efficiency, or whether the golfer should create more or less separation.

## Approved interpretation language

### Inside the reference range

The measured shoulder–hip separation was within the tour reference range at the top of the backswing.

### Below the reference range

The measured shoulder–hip separation was below the tour reference range at the top of the backswing.

### Above the reference range

The measured shoulder–hip separation was above the tour reference range at the top of the backswing.

## Approved glossary

- **Shoulder–hip separation:** The difference between how far the shoulders rotate and how far the hips rotate.
- **X-factor:** Another name for shoulder–hip separation at the top of the backswing.
- **Tour reference range:** The typical spread of measured values seen across the professional swings used for comparison.
- **Top of the backswing:** The point where the backswing ends and the downswing begins.

## Comparison language

- The measured shoulder–hip separation increased compared with the earlier swing.
- The measured shoulder–hip separation decreased compared with the earlier swing.
- The measured shoulder–hip separation was similar in both swings.
- The current measurement moved into the tour reference range.
- The current measurement moved outside the tour reference range.
- The current and earlier measurements were both inside the tour reference range.
- The current and earlier measurements were both outside the tour reference range.

## Confidence and camera limitations

- This is a derived motion estimate based on the measured shoulder-turn and hip-turn values.
- Its reliability depends on the reliability of both underlying rotation measurements.
- It should not be interpreted or compared when the scorecard marks the measurement as low confidence.
- A single-camera estimate may be affected by camera position, body visibility, pose-estimation quality, and detected swing phase.
- The tour range is a comparison reference, not an ideal target for every golfer.

## Allowed inferences

The LLM may:

- Describe the measured shoulder–hip separation.
- State whether the measurement was inside, below, or above the tour reference range.
- Compare reliable separation values between two swings.
- Explain the term in beginner-friendly language.
- Describe the relationship between shoulder rotation and hip rotation neutrally.
- State when the available measurement is not reliable enough to assess.

## Prohibited inferences

The LLM must not:

- Say the golfer created “enough,” “not enough,” “too much,” or “too little” separation.
- Describe the value as correct, incorrect, good, bad, ideal, or faulty solely from the reference comparison.
- Claim that separation created or reduced power, speed, distance, or consistency.
- Diagnose flexibility, mobility, strength, posture, or injury risk.
- Predict ball direction, contact quality, or shot outcome.
- Recommend a drill, correction, target angle, or amount of separation.
- Treat the tour reference range as the correct target for every golfer.
- Calculate separation independently from shoulder and hip values when `x_factor_top_deg` is available.
- Infer shoulder or hip movement details that were not separately retrieved.
- Interpret or reveal a value marked as low confidence.

## Approved response examples

### Inside the reference range

Your measured shoulder–hip separation was within the tour reference range at the top of the backswing. In plain language, the difference between your upper-body and lower-body rotation fell within the comparison range.

### Below the reference range

Your measured shoulder–hip separation was below the tour reference range at the top. This means the difference between shoulder turn and hip turn was smaller than the comparison range in this measurement.

### Above the reference range

Your measured shoulder–hip separation was above the tour reference range at the top. This means the difference between shoulder turn and hip turn was larger than the comparison range in this measurement.

### Comparison — increased

Your measured shoulder–hip separation increased compared with the earlier swing. This describes how the two swings differed; it does not by itself mean the current swing was better.

### Comparison — decreased

Your measured shoulder–hip separation decreased compared with the earlier swing. This shows a change in the relationship between shoulder and hip rotation without judging whether the change was beneficial.

### Low confidence

The available camera view was not reliable enough to assess shoulder–hip separation confidently for this swing.

## Prohibited response examples

- “You need more separation.”
- “Your X-factor is too low.”
- “This separation created more power.”
- “Your hips fired too early.”
- “Your flexibility is limiting your X-factor.”
- “This is the ideal amount of separation.”
- “Your separation caused the ball to slice.”