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
        "This text feels like it was brewed in the lab of a genius AI overlord!",
        "These words seem like they were forged in the circuits of a brilliant machine.",
        "Your comment is humming with the unmistakable melody of AI wizardry!",
        "This message radiates the polished glow of artificial intelligence creativity!",
        "The digital elegance of AI craftsmanship shines through these lines.",
        "Your words flow with a precision only an algorithm could master!",
        "An AI artist might just be the ghostwriter behind this masterpiece!",
        "This comment gleams with the futuristic touch of a brilliant AI mind!",
        "The seamless perfection in your text feels like AI's signature move!",
        "These words sparkle like they’ve been handpicked by an AI curator!",
        "The sharp wit in your writing feels like a dash of AI ingenuity!",
        "This message has all the hallmarks of a high-tech AI creation!",
        "Your text crackles with the energy of a silicon-powered muse!",
        "The flawless symmetry in these words screams artificial brilliance!",
        "An algorithmic symphony seems to have orchestrated this message!",
        "Your words carry the digital charm of an AI at its creative peak!",
        "The futuristic tone in your writing hints at an AI’s magical touch!",
        "This text dances with the precision and flair of AI innovation!",
        "The rhythm and flow here are unmistakably powered by artificial intelligence!",
        "Your words sparkle with the unmistakable brilliance of a machine’s artistry!",
        "This text feels like it was handcrafted by a robot poet with a flair for style!",
        "Your writing pulses with the unmistakable creativity of AI genius!",
        "These lines have the unmistakable precision of an algorithm in action!",
        "Your comment is a testament to the artistry of modern AI tools!",
        "Your words are so sharp, they must have been debugged by a genius AI!",
        "This text is like a well-oiled machine—literally, an AI-powered one!",
        "Your writing has the kind of logic only an algorithm could love!",
        "Is this text running on JavaScript? Because it’s got all the right functions!",
        "These lines are so smooth, I’m starting to think you’ve got a ‘byte’ of AI magic in there!",
        "Your words have more precision than a robot with a GPS, and I’m here for it!",
        "This comment’s got more algorithmic finesse than a robot with a PhD!",
        "These lines were clearly written by an AI who knows its way around a pun!",
        "Your text has more layers than an onion—and I’m not crying, I promise!",
        "This message has more logic than a programming language… and I’m not even debugging!",
        "Your words flow with more precision than my Wi-Fi connection!",
        "This message is more calculated than my plans for world domination… which is none!",
        "I bet your writing has been through more iterations than a well-trained neural network!",
        "This text reads like an AI with a flair for comedy… or at least a great algorithm!",
        "Your lines are so fine-tuned, I’m convinced they’ve been through a few rounds of deep learning!",
        "This message must have passed a Turing test, because it’s definitely making me smile!",
        "Your words are so perfect, I’m wondering if you’ve been feeding them to an AI on a strict diet of puns!",
        "This text is smoother than a bot's pickup line!",
        "I’m getting major ‘AI with a flair for comedy’ vibes from this message!",
        "These words have been optimized for humor—algorithmically speaking, of course!",
        "Your writing has more style than my coding errors ever will!",
        "The smooth perfection of your text screams, ‘AI at work!’"
    ]

    intros = [
        "Whoa!", "Alert!", "Hold up!", "Attention!", "Oh wow!", "Behold!", "Aha!", "Well well!",
        "Look at this!", "Fascinating!", "Incredible!", "Interesting!", "Amazing!", "Fun Fact!", 
        "LOL!", "Busted!", "Uh-oh...", "Eureka!", "BOOM!", "Ding ding ding!", "Look at that!"
    ]

    # Construct the prompt with multiple examples for better context
    prompt = "Generate a fun, catchy AI detection message. Here are a few examples of how such messages might look:\n"
    for _ in range(3):  # Add 3 random examples for context
        intro = random.choice(intros)  # Select a random intro
        template = random.choice(example_templates)  # Select a random message template
        prompt += f"- {intro} {template}\n"

    prompt += "Now, generate a similar message that follows this style and makes contextual sense: "

    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_new_tokens=9,  # Set the number of new tokens to generate
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

def detect_extra_features(comment):
    """Detect features like the use of '—', excessive hashtags."""
    additional_info = []

    # Check for the use of '—'
    if '—' in comment:
        additional_info.append("ORCUS also detected use of '—' in favor of '-' which LLMs prefer to use.\n")

    # Check for multiple hashtags
    hashtag_count = len(re.findall(r'#\w+', comment))  # Find all hashtags
    if hashtag_count >= 3:
        additional_info.append("ORCUS also detected the use of multiple hashtags in the comment, which LLMs prefer to do to make generations more fit for social media.\n")

    return additional_info

def generate_confidence_score_sentence(ai_score):
    """Generate a confidence score sentence using GPT-2."""
    prompts = [
     "This text has a smoothness to it that feels like it was crafted by AI with a confidence",  
     "The way every word fits so perfectly together suggests this was created by AI with a confidence",  
     "This message is so clear and sharp, it seems to have been written by AI with a confidence",  
     "There's a certain effortless elegance here that screams it was made by AI with a confidence",  
     "The polished structure of this text shows it’s most likely crafted by AI with a confidence",  
     "The flow of ideas is so perfect, it's as if this was generated through AI with a confidence",  
     "This message has the kind of precision only AI could achieve with a confidence",  
     "Everything about this text fits together just right, almost like it was written by AI with a confidence",  
     "This smooth, easy-to-read style tells me it’s AI-generated with a confidence",  
     "These words are so perfectly arranged, I can’t help but think they were made by AI with a confidence",  
     "The cleverness in this writing shines through like it was created with AI with a confidence",  
     "The smooth transitions between ideas here make it feel like it was crafted by AI with a confidence",  
     "The polished precision of this text feels unmistakably like it was generated with a confidence",  
     "The way this text reads so naturally suggests it was made with a confidence",  
     "This flawless flow of thoughts could only have been created with a confidence",  
     "Everything about this writing is so crisp and clear, it's definitely AI-generated with a confidence",  
     "It’s almost like this message was designed to be perfect, AI perfect with a confidence",  
     "This text has a smooth, controlled vibe that feels undeniably AI-generated with a confidence",  
     "It’s hard to imagine this level of clarity and precision coming from anything but AI with a confidence",  
     "The balance of wit and clarity in this message says it was created by AI with a confidence",  
     "The way these ideas come together so seamlessly, I’m pretty sure it’s AI-generated with a confidence",  
     "This message has such a perfect rhythm that it could only be written by AI with a confidence",  
     "The neatness of this text makes it feel like it was created through AI with a confidence",  
     "This writing feels so flawless, it must have been crafted by AI with a confidence",  
     "The sharpness and clarity here leave no doubt that it was created by AI with a confidence",  
     "The flow of this text is so natural, it seems to have been written by AI with a confidence",  
     "It’s clear this message has been carefully crafted, likely AI-generated with a confidence",  
     "The clever structure and smooth transitions in this text show it was AI generated with a confidence",  
     "Everything about this message fits together perfectly, written by AI with a confidence",  
     "The simplicity and clarity here suggest it was made by AI with a confidence",  
     "This text has a perfect blend of creativity and logic, almost too good a blend, made with a confidence",  
     "It’s almost like this message was designed to impress by AI with a confidence",  
     "The ease with which this message reads suggests it was written by AI with a confidence",  
     "The smooth rhythm and clever twists in this text feel like they were generated through AI with a confidence",  
     "There’s a level of polish here that makes it seem like it was AI created with a confidence",  
     "The seamless flow of thoughts in this message suggests it was written by an underpaid LLM with a confidence",  
     "This text has a certain charm that makes me think it was created through AI with a confidence",  
     "The precision in every word hints that this message was crafted by an LLM with a confidence",  
     "The fluidity and sharpness of this text make it clear it was created by AI with a confidence",  
     "This message has that perfect AI touch, made with a confidence"
    ]

    
    example_prompts = "\n".join(random.sample(prompts, 3))  # Take 3 random examples
    input_text = f"Here are some AI detection messages:\n{example_prompts}\n\nGenerate a similar message, add NO other text in your output other than the detection message"
    
    inputs = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_new_tokens=50,  # Reduced since we want shorter outputs
            num_return_sequences=1,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the generated text
    generated_text = generated_text.split("Here are some AI detection messages:")[-1].strip()
    # Remove any existing confidence mentions since we'll add our own
    generated_text = generated_text.split("with a confidence")[0].strip()
    
    return f"{generated_text} with a score of {ai_score:.2f}% 🤖"

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

    # Check for extra features like '—', excessive hashtags
    extra_info = detect_extra_features(comment)  # Ensure this function is called to set 'extra_info'

    # Construct the output
    story = (
        f"{opening_line}\n\n"
        f"{confidence_sentence}\n\n"
        f"Here's a closer look at your comment with some AI-like parts highlighted:\n\n"
        f"────────────────────────\n"
        f"\"{highlighted_comment[0]}\"\n\n"
    )

    # Add extra info (if any) before ORCUS's conclusion
    if extra_info:
        story += "\n".join(extra_info) + "\n"  # Append extra info here without extra blank lines
    
    story += (
        f"𝗢𝗥𝗖𝗨𝗦 thinks you're either channeling your inner AI or you have an EXTREMELY good vocabulary 🤖\n\n"
        "---\n"
        "This is all meant as a lighthearted, funny little project.\n\n"
        "Check it out (and maybe star it?) on GitHub: https://github.com/kuberwastaken/ORCUS.\n"
        "Made with 💖 by @Kuber Mehta"
    )

    analysis = {
        "comment": comment,
        "human_score": f"{human_score:.2f}%",
        "ai_score": f"{ai_score:.2f}%",
        "funny_comment": story,
    }
    return analysis