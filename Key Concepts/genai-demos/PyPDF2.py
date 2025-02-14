# pip install PyPDF2
import PyPDF2

# Open the PDF file in binary mode
pdf_file = open('example.pdf', 'rb')

# Create a PDF reader object
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# Extract text from the PDF file
text = ''
for page_num in range(pdf_reader.getNumPages()):
    page = pdf_reader.getPage(page_num)
    text += page.extractText()

# Create a text file and write the extracted text to it
text_file = open('example.txt', 'w', encoding='utf-8')
text_file.write(text)
text_file.close()

# Extract metadata from the PDF file
metadata = pdf_reader.getDocumentInfo()

# Print the extracted text and metadata
print("Extracted Text:\n", text)
print("\nMetadata:")
for key, value in metadata.items():
    print(key, ":", value)
    
# Close the PDF file
pdf_file.close()