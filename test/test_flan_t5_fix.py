"""Quick test to verify FLAN-T5 model is working correctly."""

from transformers import pipeline

print("Testing FLAN-T5 model...")
print("=" * 70)

# Load the model
print("\n1. Loading FLAN-T5 from local directory...")
pipe = pipeline(
    "text2text-generation",
    model="./model/flan-t5-small",
    device="mps"
)
print("✓ Model loaded successfully")

# Test with a few examples
print("\n2. Testing sentiment classification...")
test_cases = [
    "Is the following sentence positive or negative? This movie is absolutely terrible and boring.",
    "Is the following sentence positive or negative? This is a wonderful and amazing film!",
    "Is the following sentence positive or negative? The acting was great but the plot was weak.",
]

for i, prompt in enumerate(test_cases, 1):
    result = pipe(prompt, max_length=10)[0]["generated_text"]
    print(f"\n   Test {i}:")
    print(f"   Input: {prompt[:70]}...")
    print(f"   Output: '{result}'")

print("\n" + "=" * 70)
print("✓ Test complete!")
print("\nIf you see 'negative' and 'positive' in the outputs above,")
print("the model is working correctly!")
