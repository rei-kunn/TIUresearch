import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure necessary NLTK datasets are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Lemmatizer
wordlemmatizer = WordNetLemmatizer()
Stopwords = set(stopwords.words('english'))

def process_claims_v10(dirty_claim):
    # Split the claims on each claim number or range of claim numbers
    split_claims = re.split(r'(?<=\.\s)(\d+-*\d*\.\s)', dirty_claim)

    combined_sentences = []
    current_claim = ""

    for part in split_claims:
        if re.match(r'\d+-*\d*\.\s', part):
            # If current_claim is not empty, add it to combined_sentences
            if current_claim:
                combined_sentences.append(current_claim.strip())
            current_claim = part
        else:
            current_claim += part

    # Add the last claim
    if current_claim:
        combined_sentences.append(current_claim.strip())

    # Handle ranges of canceled claims and keep the subsequent claims
    processed_claims = []
    for claim in combined_sentences:
        if "-" in claim and "(canceled)" in claim:
            start, end = [int(num) for num in re.findall(r'(\d+)-(\d+)', claim)[0]]
            for i in range(start, end + 1):
                processed_claims.append(f"{i}. (canceled)")
            # Append the text following the canceled range, if any
            following_text = claim.split(")")[1]
            if following_text.strip():
                processed_claims.append(following_text.strip())
        else:
            processed_claims.append(claim)

    return processed_claims


def extract_dependencies(claims):
    dependencies = {}
    for i, claim in enumerate(claims):
        claim_number = i + 1  # Assuming claim numbers are sequential and start at 1
        dependencies[claim_number] = []

        # Check for specific claim references
        for ref_match in re.finditer(r'claim (\d+)', claim):
            referenced_claim = int(ref_match.group(1))
            dependencies[claim_number].append(referenced_claim)

        # Check for general references to preceding claims
        if "any of the preceding claims" in claim:
            # Add all preceding claim numbers as dependencies
            dependencies[claim_number].extend(range(1, claim_number))

    return dependencies