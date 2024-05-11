import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    sentences = sent_tokenize(text)

    # Tokenize words in each sentence
    words = [word_tokenize(sentence) for sentence in sentences]

    # Flatten the list of words
    words = [word for sentence_words in words for word in sentence_words]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return sentences, lemmatized_words

def extract_questions(preprocessed_text):
    questions = []

    interrogative_words = ["who", "what", "where", "when", "why", "how", "which", "whose", "whom", "explain", "tell", "do", "did", "question", "questions", "find", "write", "code", "programming", "approach"]

    for sentence in preprocessed_text:
        words = word_tokenize(sentence)  # Tokenize the sentence into words
        if any(word.lower() in interrogative_words or word.endswith('?') for word in words):
            questions.append(sentence)

    return questions


"""
# Example overall experience text
text = "The interview started with self introduction. There was a panel of 2 people , one person concentrated on resume and the other person was into DSA questions. The first interviewer started asking me what ever i have mentioned in my resume and in my introduction, then a thorough discussion about my internship and the workflow. Followed by some dbms questions which were basic and some questions related to javascript , react js and Node js as i have mentioned them in my resume. Then the second interviewer gave me 2 dsa questions and asked me to share my screen and write code on notepad. 1. Implement stack using queue. 2. Find the maximum number in an array. 3.  Follow up question is to find the second highest number in the array. what is array?. Overall experience was good. questions were mainly from data structures and algorithms. mostly focused on problem solving skills and approach towards the problem. Also in depth questions about the projects mentioned. Asked  to code a few data structures questions.topics : strings, subsequences.Trie data structure. finally some hr questions like, what do you know about airtel?  why airtel? totally the experience was good and interview was challenging. The aptitude test was easy. The test had 5 sections. I couldn't clear this round. it all depends on luck. Not all students who gave the test good were selected. selection process was bad. overall experience was bad and terrible. The interviewer was very friendly and asked me to relax and think of it as an informal talk. He asked me: Tell me about yourself. Asked me further questions from what all I mentioned in my introduction. Then he asked me some questions about Cloud since I had applied for the BO-Cloud role. What do you know about Cloud? What are the different cloud service and cloud deployment models? What are your aspirations in life? To which I had initially replied something related to advancing in my career, then he asked me to specify my personal aspirations instead of academic or career-oriented aspirations. Then the HR round:  I cleared this round. Introduce Yourself. Why Deloitte? Where do you see yourself in 5 years? I did not clear this round.  My interview primarily comprised HR inquiries, while some of my acquaintances encountered a blend of HR and technical assessments. The interviewer fostered a comfortable environment, initiating our interaction with a brief self-introduction and extending an invitation for any queries I might have. Lasting approximately 25 minutes, the interview centred on aspects of my project work. Specific inquiries delved into the uniqueness of my project, clarifying whether it was a collaborative effort or an individual endeavour. Given that my project was a group endeavour, I fielded questions regarding the challenges encountered while collaborating with team members and the obstacles faced during project execution. Additionally, discussions touched upon personal strengths and weaknesses. The interviewer expressed satisfaction with my responses and concluded the session on a positive note, remarking that it was a pleasure conversing with me. MS Office and Common applications, Pseudo Code and Fundamentals of Cloud and Network security To qualify for this round, one needs to clear the sectional as well as the sub-sectional cutoff. After clearing this round, there was a Coding round, which consisted of 2 questions to be solved in 45 minutes. The first question was finding the Sum of odds and evens in a number and the second was Max product of multidimensional array. The questions were different for different people but the difficulty was overall the same. I’ve solved one question. After a few days, the link for the Communication assessment was sent, which was the Pearson Versant test and was around 20 minutes. The last round was the Virtual Interview, which had 2 members in the panel. The questions asked were on the projects and basic HR questions. Some questions are: Introduce Yourself. Tell us about your Projects. Your Role in the Project. What are the issues you faced while doing the Project? Did you face any issues with your team members? If you have a chance to extend your project. What would it be? Do u have any questions for us? After a few days, I got a mail from Accenture that I was selected for the Associate Software Engineer role.  First I was asked to show my gov id proof. This the only round of interviews which consists of 3 panelists namely the TR(Technical), MR(Managerial), and HR.  Technical Round: Tell me about Yourself. Tell me about Project. What programming languages do you know? Difference Between C and C++. What are the things that make C++ an object-oriented programming language? What is the OOPS concept? What is RDBMS? What is SQL and the Full Form of SQL? Definition of DBMS and Advantages. Advantages of Python language. C++ is platform-independent or not. Why Python is easy? What is a list in python? What are tuples in python? What is machine learning? Explain Supervised and Unsupervised learning with example. What is Classification and Where we can apply to explain with example? HR Round: What are your strengths and weaknesses? How You overcome your Weakness? You Enjoyed your college life or not? Why should I hire you? What are your hobbies? Are you motivated? What makes you happy? Are you ready to relocate? Tell me about TCS & Why you want to join TCS? If we assigned a location in Chennai to you then after will you accept a job?"

# Preprocess text
sentences, preprocessed_words = preprocess_text(text)
print(sentences)
# Extract questions
questions = extract_questions(sentences)

print("Questions extracted from the overall experience:")
for idx, question in enumerate(questions, start=1):
    print(f"{idx}. {question}")
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Dataset Preparation
data = pd.read_csv("D:\Downloaded Files\question1.csv")  # Load the dataset

# Step 2: Data Preprocessing
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters, numbers, and punctuations
    return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

data['cleaned_text'] = data['Question'].apply(clean_text)  # Clean text
data['tokens'] = data['cleaned_text'].apply(tokenize_and_lemmatize)  # Tokenize and lemmatize

# Step 3: Label Encoding
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Category'])

# Step 4: Model Architecture
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Step 5: Training and Evaluation
max_len = 128  # Max length of input tokens

# Train-validation split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenize input text and convert to input IDs
def tokenize_data(data):
    input_ids = []
    attention_masks = []

    for question in data['cleaned_text']:
        encoded_dict = tokenizer.encode_plus(
                            question,
                            add_special_tokens=True,
                            max_length=max_len,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(data['label'].values)
    labels = labels.to(torch.long)

    return TensorDataset(input_ids, attention_masks, labels)

train_dataset = tokenize_data(train_data)
val_dataset = tokenize_data(val_data)

batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Step 6: Model Training and Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    model.eval()
    val_loss, val_accuracy = 0, 0
    num_val_steps, num_val_examples = 0, 0

    for batch in tqdm(val_dataloader, desc=f"Validation"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].to('cpu').numpy()

        val_loss += outputs.loss.item()
        val_accuracy += (logits.argmax(axis=1) == label_ids).mean()
        num_val_steps += 1

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_accuracy = val_accuracy / len(val_dataloader)

    print(f"Epoch {epoch + 1}:")
    print(f"  Training Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")
    print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")

# Step 7: Fine-tuning
# You can fine-tune hyperparameters by adjusting learning rate, batch size, etc.

# Step 8: Inference
# Use the trained model to predict categories of new questions
# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert_model")
# Save the label encoder
#tokenizer.save("fine_tuned_bert_tokenizer")
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

