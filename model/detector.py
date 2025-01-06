from transformers import RobertaForSequenceClassification, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import random

# Load the models and tokenizers
MODEL_NAME = "roberta-base-openai-detector"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)

# Load GPT-2 model and tokenizer
gpt2_model_name = "openai-community/gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

def highlight_most_ai_like_phrases(comment, top_k=1):
    """Highlight the most AI-like phrases."""
    sentences = re.split(r'(?<=[.!?]) +', comment)
    scores = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            ai_score = torch.softmax(logits, dim=1).tolist()[0][1]
            scores.append((sentence, ai_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [sentence for sentence, _ in scores]

def generate_dynamic_opening():
    """Generate a dynamic, catchy opening line with emojis using GPT-2."""
    example_templates = [
        "Your text feels like it's from another dimension where AI rules supreme!",
        "These words seem to be crafted by an artificial mind in the digital realm.",
        "This message appears to have a vibe that’s both digital and human!",
        "Your writing resonates with the unmistakable signature of artificial intelligence!",
        "The digital fingerprints of AI are all over this text.",
        "Your words dance with the distinct rhythm of artificial intelligence!",
        "An AI seems to have channeled its creativity through these words.",
        "The algorithmic beauty of AI shines through your text!",
        "Your message carries the distinctive mark of AI craftsmanship.",
        "This text sparkles with artificial intelligence brilliance but it’s definitely got personality!",
        "Your writing exhibits the telltale signs of AI artistry",
        "The mathematical precision of AI echoes in your words!"
    ]
    
    intros = [
        "Whoa!",
        "Alert!",
        "Hold up!",
        "Attention!",
        "Oh wow!",
        "Behold!",
        "Aha!",
        "Well well!",
        "Look at this!",
        "Fascinating!",
        "Incredible!",
        "Interesting!",
        "Amazing!"
    ]
    
    # Construct the prompt with multiple examples for better context
    prompt = "Generate fun, and catchy AI detection messages. Examples:\n"
    for _ in range(3):  # Add 3 random examples for context
        prompt += f"- {random.choice(intros)} {random.choice(example_templates)}\n"
    prompt += "Generate a catchy, fun message with some AI flair that’s not too long: "
    
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_new_tokens=70,  # Set the number of new tokens to generate
            num_return_sequences=1,
            temperature=0.85,  # Slightly higher for creativity, but not too much
            top_k=50,
            top_p=0.92,  # Keep diversity but limit to the best options
            no_repeat_ngram_size=2,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the generated text
    generated_text = generated_text.split("Generate a catchy, fun message with some AI flair that’s not too long:")[-1].strip()
    generated_text = generated_text.split("\n")[0].strip()  # Take only the first line
    
    # If generation went off track, use template
    if (len(generated_text.split()) > 15 or 
        any(word in generated_text.lower() for word in ["news", "reuters", "reported", "according"]) or
        len(generated_text.split()) < 5):
        intro = random.choice(intros)
        template = random.choice(example_templates)
        generated_text = f"{intro} {template}"
    
    # Add emojis
    emojis = ["⚡️", "🌌", "🔮", "🚀", "💫", "✨", "🤖", "🚨"]
    start_emoji = random.choice(emojis)
    end_emoji = start_emoji
    
    # Ensure proper punctuation
    if not generated_text.endswith(("!", ".", "...")):
        generated_text += "!"
    
    return f"{start_emoji} {generated_text} {end_emoji}"

def generate_confidence_score_sentence(ai_score):
    """Generate a confidence score sentence using GPT-2."""
    prompts = [
        "Alert! This text appears to be AI generated with confidence",
        "Hold on! This message seems AI generated with confidence",
        "Wow! This text has been detected as AI with confidence",
        "Guess what! This content appears to be AI with confidence"
    ]
    
    prompt = random.choice(prompts)
    input_text = f"Generate a detection alert: {prompt}"
    
    inputs = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_length=40,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the generated text and format with the confidence score
    generated_text = generated_text.split("Generate a detection alert: ")[-1].strip()
    generated_text = generated_text.split("confidence")[0].strip()
    
    return f"{generated_text} has been flagged as 𝗔𝗜 𝗚𝗘𝗡𝗘𝗥𝗔𝗧𝗘𝗗 with a confidence score of {ai_score:.2f}% 🤖"

def analyze_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]

    human_score = probabilities[0] * 100
    ai_score = probabilities[1] * 100
    highlighted_comment = highlight_most_ai_like_phrases(comment)

    # Generate opening line and confidence score sentence using GPT-2
    opening_line = generate_dynamic_opening()
    confidence_sentence = generate_confidence_score_sentence(ai_score)

    # Construct the output
    story = (
        f"{opening_line}\n\n"
        f"────────────────────────\n\n"
        f"{confidence_sentence}\n\n"
        f"Here's a closer look at your comment with some AI-like parts highlighted:\n\n"
        f"\"{highlighted_comment[0]}\"\n\n"
        f"𝗢𝗥𝗖𝗨𝗦 thinks you're either channeling your inner AI or you have an EXTREMELY good vocabulary 🤖\n\n"
        "---\n"
        "This is all meant as a lighthearted, funny little project.\n\n"
        "Check it out on GitHub: https://github.com/kuberwastaken/ORCUS.\n"
        "Made with 💖 by @Kuber Mehta"
    )

    analysis = {
        "comment": comment,
        "human_score": f"{human_score:.2f}%",
        "ai_score": f"{ai_score:.2f}%",
        "funny_comment": story,
    }
    return analysis