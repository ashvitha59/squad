import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import random

# Expanded language pairs for translation
LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Arabic": "ar",
    "Turkish": "tr",
    "Swedish": "sv",
    "Greek": "el",
    "Polish": "pl",
    "Danish": "da",
    "Finnish": "fi",
    "Hebrew": "he",
    "Indonesian": "id",
    "Thai": "th",
    "Vietnamese": "vi"
}

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the M2M100 model and tokenizer for multilingual translation."""
    model_name = "facebook/m2m100_418M"
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def translate_text(text, src_lang, tgt_lang):
    """Translate the given text from source to target language using M2M100."""
    tokenizer, model = load_model()
    if tokenizer and model:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    else:
        return "Translation failed. Please check the model or input."

# ------------------------------------------
# 🚀 Login Page
# ------------------------------------------
def login_page():
    st.title("🔐 User Login")
    
    # Input fields
    first_name = st.text_input("First Name")
    middle_name = st.text_input("Middle Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")

    # Validate input
    if st.button("Submit"):
        if first_name and last_name and email:
            st.session_state.logged_in = True
            st.session_state.user_info = {
                "first_name": first_name,
                "middle_name": middle_name,
                "last_name": last_name,
                "email": email
            }
            st.rerun()  # ✅ Use `st.rerun()` instead of `st.experimental_rerun()`
        else:
            st.error("Please fill in all required fields.")

# ------------------------------------------
# 🌍 Main Application
# ------------------------------------------
def main_app():
    st.sidebar.title(f"👋 Welcome, {st.session_state.user_info['first_name']}!")
    
    # Logout Button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()  # ✅ Use `st.rerun()` instead of `st.experimental_rerun()`

    # Sidebar Navigation
    option = st.sidebar.selectbox("Choose an option:", ["Translation", "Note Taking", "Word Guessing Game"])

    # ------------------------------------------
    # 1️⃣ Language Translation Section
    # ------------------------------------------
    if option == "Translation":
        st.subheader("Translate Text")

        text_input = st.text_area("Enter text to translate:", height=150)
        src_lang = st.selectbox("Select Source Language:", list(LANGUAGE_CODES.keys()))
        tgt_lang = st.selectbox("Select Target Language:", list(LANGUAGE_CODES.keys()))

        if st.button("Translate"):
            if text_input.strip():
                translation = translate_text(text_input, LANGUAGE_CODES[src_lang], LANGUAGE_CODES[tgt_lang])
                st.success("Translated Text:")
                st.write(translation)
            else:
                st.warning("Please enter text to translate.")

    # ------------------------------------------
    # 2️⃣ Note-Taking Section
    # ------------------------------------------
    elif option == "Note Taking":
        st.subheader("Notes")
        note = st.text_area("Write your notes here:", height=150)

        if "notes" not in st.session_state:
            st.session_state.notes = []

        if st.button("Save Note"):
            if note.strip():
                st.session_state.notes.append(note)
                st.success("Note saved!")
            else:
                st.warning("Cannot save an empty note.")

        # Display saved notes
        if st.session_state.notes:
            st.subheader("Saved Notes")
            for idx, saved_note in enumerate(st.session_state.notes, start=1):
                st.write(f"{idx}. {saved_note}")

    # ------------------------------------------
    # 3️⃣ Word Guessing Game (English to French)
    # ------------------------------------------
    elif option == "Word Guessing Game":
        st.subheader("Word Guessing Game 🎮")
        st.markdown("### *Translation from English to French*")

        # Sample words dataset
        words = {
            "hello": "bonjour",
            "goodbye": "au revoir",
            "thank you": "merci",
            "please": "s'il vous plaît",
            "love": "amour",
            "friend": "ami",
            "family": "famille",
            "happy": "heureux",
            "sad": "triste",
            "food": "nourriture",
            "water": "eau",
            "book": "livre",
            "music": "musique",
            "sun": "soleil",
            "moon": "lune"
        }

        # Initialize session state for tracking game progress
        if "current_word" not in st.session_state:
            st.session_state.current_word = random.choice(list(words.keys()))
        if "checked" not in st.session_state:
            st.session_state.checked = False

        current_word = st.session_state.current_word
        correct_translation = words[current_word]

        st.write(f"Translate this word into *French: **{current_word}*")

        user_input = st.text_input("Your answer:")

        if st.button("Check"):
            if user_input.lower().strip() == correct_translation.lower():
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect! The correct answer is: *{correct_translation}*")
            st.session_state.checked = True

        if st.session_state.checked:
            if st.button("Next"):
                st.session_state.current_word = random.choice(list(words.keys()))
                st.session_state.checked = False
                st.rerun()  # ✅ Use `st.rerun()` instead of `st.experimental_rerun()`

# ------------------------------------------
# 🏁 App Entry Point
# ------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
