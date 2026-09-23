

"""Text Splitters and Chunking Strategies Optimizing document chunks for RAG"""

from langchain_text_splitters import (
RecursiveCharacterTextSplitter,
CharacterTextSplitter,
TokenTextSplitter,
MarkdownHeaderTextSplitter,
Language,
)
from langchain_core.documents import Document
from dotenv import load_dotenv


# RecursiveCharacterTextSplitter: Splits text by a list of characters (like newlines and spaces) recursively to keep paragraphs and sentences together.Preserves semantic coherence.
# CharacterTextSplitter: Splits text based on a single, user-defined separator character.
# TokenTextSplitter: Measures chunk size by the number of tokens rather than raw character counts.
# MarkdownHeaderTextSplitter: Splits text based on specified Markdown headers (e.g., #, ##).
# Language: An enum used to specify code structures when splitting programming languages.Language is a class/enum-like collection of predefined programming languages that LangChain's code splitter understands.
# Document: The base class used in LangChain to represent text content along with its associated metadata.


# Markdown Headers are: 
## Main Heading
## Second-Level Heading
### Third-Level Heading
#### Fourth-Level Heading


#Both Document and each chunk are Document objects,LangChain creates multiple smaller Document objects,each chunk is itself a LangChain Document object. so they both have:
#metadata
#page_content


SAMPLE_TEXT="""
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        elif arr[mid] < target:
            left = mid + 1

        else:
            right = mid - 1

    return -1


numbers = [2, 5, 8, 12, 16, 23, 38, 45, 56, 72]

target = 23

result = binary_search(numbers, target)

if result != -1:
    print(f"Target found at index {result}")
else:
    print("Target not found")


for number in numbers:
    print("Number:", number)


"""

python_splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50    
)

chunks=python_splitter.split_text(SAMPLE_TEXT)

print(f"Code Splitter produced {len(chunks)} chunks.")
for i, chunk in enumerate (chunks):
    print(f"\nChunk {i} ({len (chunk)} chars):")
    print(chunk[:150] + "..." if len(chunk) > 150 else chunk)




