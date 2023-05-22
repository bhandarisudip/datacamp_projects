'''
When you call nlp on a string, spaCy
first tokenizes the text and creates 
a document object. In this exercise, 
you'll learn more about the Doc, as well 
as its views Token and Span.
'''

'''
Import the English language class and create the nlp object.
Process the text and instantiate a Doc object in the variable doc.
Select the first token of the Doc and print its text.
'''
# Import the English language class and create the nlp object
from spacy.lang.en import English
nlp = English()

# Process the text
doc = nlp("I like tree kangaroos and narwhals.")

# Select the first token
first_token = doc[0]

# Print the first token's text
print(first_token.text)


'''
Create a slice of the Doc for the tokens 
"tree kangaroos" and "tree kangaroos and narwhals".
'''

# Import the English language class and create the nlp object
from spacy.lang.en import English
nlp = English()

# Process the text
doc = nlp("I like tree kangaroos and narwhals.")

# A slice of the Doc for "tree kangaroos"
tree_kangaroos = doc[2:4]
print(tree_kangaroos.text)

# A slice of the Doc for "tree kangaroos and narwhals" (without the ".")
tree_kangaroos_and_narwhals = doc[2:6]
print(tree_kangaroos_and_narwhals.text)
