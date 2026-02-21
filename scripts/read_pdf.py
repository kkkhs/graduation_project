import pypdfium2

def read_pdf(pdf_path):
    pdf = pypdfium2.PdfDocument(pdf_path)
    text = ""
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        page_text = page.get_textpage().get_text_range()
        text += f"\n\n{'='*80}\n"
        text += f"Page {page_num + 1}\n"
        text += f"{'='*80}\n"
        text += page_text
    pdf.close()
    return text

if __name__ == "__main__":
    pdf_path = r"E:\Codes\Githubs\graduation_project\pdfs\2022_jianqi_chen_a.pdf"
    text = read_pdf(pdf_path)
    
    output_path = r"E:\Codes\Githubs\graduation_project\pdfs\2022_jianqi_chen_a.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"PDF text extracted to: {output_path}")
    print(f"Total characters: {len(text)}")
